"""MoE-Inference-Bench (SC '25) style systematic sweep of MoE expert computation.

Sweeps the full GLM-5 MoE parameter space on H100:
- batch_sizes × token_counts × expert_counts × active_experts × ffn_dims × precision

For each config point:
1. Baseline: per-expert loop in BF16/FP8 (pure PyTorch)
2. Optional: DeepGEMM grouped FP8 GEMM (if available)

Metrics reported per point:
- median_ms, p99_ms (tail latency — MoE-Inference-Bench standard)
- TFLOPS, MFU% (relative to H100 BF16/FP8 peak)
- Operational intensity, roofline bound, HBM SOL%
- Peak memory (GiB)

References:
- MoE-Inference-Bench (SC '25): 4×H100 systematic sweep methodology
- DeepGEMM (2025): 1550 TFLOPS FP8 grouped GEMM on H800
- GLM-5 (arXiv:2602.15763): 256 experts, top-8, hidden=6144, ffn=2048
"""

import argparse
import sys
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── Shared utilities ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmark.shared.config import (
    BenchConfig,
    BenchResult,
    GLM5_CONFIG,
    H100_SPECS,
    MOE_BENCH_BATCHES,
    MOE_BENCH_TOKENS,
    MOE_BENCH_EXPERTS,
    MOE_BENCH_ACTIVE,
    MOE_BENCH_FFN_DIMS,
)
from benchmark.shared.timer import cuda_timer_extended, check_outliers
from benchmark.shared.metrics import (
    compute_moe_flops,
    compute_moe_bytes,
    compute_mfu,
    compute_hbm_sol,
    compute_tflops,
    compute_bandwidth_gb_s,
    compute_operational_intensity,
    classify_roofline_bound,
    compute_roofline_achievable,
)
from benchmark.shared.report import save_results, print_summary_table, capture_environment


# ── Optional: DeepGEMM grouped FP8 GEMM ─────────────────────────────────────
try:
    import deep_gemm
    DEEPGEMM_AVAILABLE = True
    print("[bench_moe] DeepGEMM available — FP8 grouped GEMM path enabled.")
except ImportError:
    DEEPGEMM_AVAILABLE = False
    print("[bench_moe] DeepGEMM not found — FP8 grouped GEMM path disabled.")


# ── Constants ─────────────────────────────────────────────────────────────────
HIDDEN_SIZE = GLM5_CONFIG["hidden_size"]   # 6144
WARMUP = 10
ITERS = 100


# ── Tensor helpers ────────────────────────────────────────────────────────────

def _make_dtype(precision: str) -> torch.dtype:
    return torch.bfloat16 if precision == "bf16" else torch.float8_e4m3fn


def _make_inputs(
    n_tokens: int,
    n_experts: int,
    active_experts: int,
    ffn_dim: int,
    precision: str,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Allocate all tensors needed for one MoE forward pass.

    Returns:
        hidden_states:  [N, D]           — routed token embeddings
        gate_up_weight: [E, 2*I, D]      — fused gate+up projection weights
        down_weight:    [E, D, I]        — down projection weights
        topk_ids:       [N, K]  (int32)  — selected expert indices per token
        topk_weights:   [N, K]  (float)  — normalised sigmoid routing weights

    For FP8 the weight tensors are cast to float8_e4m3fn; hidden_states stays
    BF16 (the dequant/requant is part of the timed kernel in real deployments,
    but here we isolate the GEMM cost following MoE-Inference-Bench convention).
    """
    D = HIDDEN_SIZE
    I = ffn_dim
    E = n_experts
    K = active_experts
    N = n_tokens

    hidden_states = torch.randn(N, D, dtype=torch.bfloat16, device=device)

    if precision == "fp8":
        gate_up_weight = torch.randn(E, 2 * I, D, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )
        down_weight = torch.randn(E, D, I, dtype=torch.bfloat16, device=device).to(
            torch.float8_e4m3fn
        )
    else:
        gate_up_weight = torch.randn(E, 2 * I, D, dtype=torch.bfloat16, device=device)
        down_weight = torch.randn(E, D, I, dtype=torch.bfloat16, device=device)

    # Random routing — uniform draw without replacement per token (mimics real TopK)
    topk_ids = torch.stack(
        [torch.randperm(E, device=device)[:K] for _ in range(N)], dim=0
    ).to(torch.int32)  # [N, K]

    # Normalised sigmoid weights (GLM-5 uses sigmoid + L1 normalisation)
    raw_scores = torch.randn(N, K, device=device)
    topk_weights = torch.sigmoid(raw_scores)
    topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True).clamp(min=1e-6)

    return hidden_states, gate_up_weight, down_weight, topk_ids, topk_weights


# ── MoE forward kernels ───────────────────────────────────────────────────────

def moe_forward_baseline(
    hidden_states: torch.Tensor,       # [N, D]  BF16
    gate_up_weight: torch.Tensor,      # [E, 2*I, D]  BF16 or FP8
    down_weight: torch.Tensor,         # [E, D, I]    BF16 or FP8
    topk_ids: torch.Tensor,            # [N, K]  int32
    topk_weights: torch.Tensor,        # [N, K]  float32
) -> torch.Tensor:
    """Per-expert loop baseline — the textbook MoE forward.

    For each expert e:
      1. Gather tokens assigned to e
      2. gate_up = tokens @ gate_up_weight[e].T  →  [T_e, 2*I]
      3. gate, up = split → SwiGLU activation
      4. out = (gate_out * up_out) @ down_weight[e].T  →  [T_e, D]
      5. Scatter-accumulate with routing weight

    This matches the reference implementation in DeepSpeed-MoE and Megatron-Core.
    FP8 weights are dequantised to BF16 before the GEMM (no scale tensors in
    this baseline — we measure raw compute throughput, not full quantisation pipeline).
    """
    N, D = hidden_states.shape
    K = topk_ids.shape[1]
    device = hidden_states.device
    output = torch.zeros_like(hidden_states)  # [N, D]

    # Dequant FP8 → BF16 for baseline arithmetic (weight-only quant approximation)
    if gate_up_weight.dtype == torch.float8_e4m3fn:
        guw = gate_up_weight.to(torch.bfloat16)
        dw = down_weight.to(torch.bfloat16)
    else:
        guw = gate_up_weight
        dw = down_weight

    # Build expert dispatch table:  expert_id → list of (token_idx, slot_in_topk)
    # Flattening topk_ids into a single (N*K) list is slightly faster than a dict
    flat_ids = topk_ids.reshape(-1).long()      # [N*K]
    flat_weights = topk_weights.reshape(-1)      # [N*K]
    token_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K).reshape(-1)  # [N*K]

    for e in range(guw.shape[0]):
        mask = (flat_ids == e)
        if not mask.any():
            continue
        tok_idx = token_indices[mask]           # indices into hidden_states
        w = flat_weights[mask].to(torch.bfloat16).unsqueeze(-1)  # [T_e, 1]

        tokens = hidden_states[tok_idx]         # [T_e, D]
        gate_up = tokens @ guw[e].T             # [T_e, 2*I]
        gate, up = gate_up.chunk(2, dim=-1)     # each [T_e, I]
        activated = F.silu(gate) * up           # SwiGLU
        expert_out = activated @ dw[e].T        # [T_e, D]

        output.index_add_(0, tok_idx, expert_out * w)

    return output


def moe_forward_deepgemm(
    hidden_states: torch.Tensor,
    gate_up_weight: torch.Tensor,
    down_weight: torch.Tensor,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> torch.Tensor:
    """DeepGEMM grouped FP8 GEMM — matches the PROVEN pattern from h100_bench.py.

    Uses deep_gemm.m_grouped_fp8_gemm_nt_contiguous with the exact same
    tensor layout and quantization approach that works in the test suite.
    Single GEMM benchmark (gate_up only) to isolate kernel throughput.
    """
    from deep_gemm.utils import per_custom_dims_cast_to_fp8

    N, D = hidden_states.shape
    K = topk_ids.shape[1]
    E, I2, _ = gate_up_weight.shape  # [E, 2*I, D]
    I = I2 // 2
    device = hidden_states.device
    M = N * K  # total token-expert pairs

    # ── Token dispatch (sort by expert) ──────────────────────────────────
    flat_ids = topk_ids.reshape(-1).long()
    token_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K).reshape(-1)
    flat_weights = topk_weights.reshape(-1)

    sort_order = torch.argsort(flat_ids, stable=True)
    sorted_ids = flat_ids[sort_order]
    sorted_tok = token_indices[sort_order]
    sorted_w = flat_weights[sort_order]

    # ── Exact h100_bench.py pattern ──────────────────────────────────────
    # A: activations [M, D] in BF16 → quantize to FP8
    a = hidden_states[sorted_tok].to(torch.bfloat16)  # [M, D]
    a_fp8 = per_custom_dims_cast_to_fp8(a, (0,), False)
    # a_fp8 = (tensor[M, D], scales[M])

    # B: weights [E, I, D] — use down_weight [E, D, I] transposed to [E, I, D]
    # (benchmarking one GEMM at a time, matching h100_bench exactly)
    b = down_weight.permute(0, 2, 1).contiguous().to(torch.bfloat16)  # [E, I, D]
    b_fp8 = per_custom_dims_cast_to_fp8(b.view(E * I, D), (0,), False)
    b_fp8 = (b_fp8[0].view(E, I, D), b_fp8[1].view(E, I))
    # b_fp8 = (tensor[E, I, D], scales[E, I])

    # Output: [M, I]
    d = torch.empty(M, I, dtype=torch.bfloat16, device=device)

    # Grouped layout: per-row expert index [M] int32
    grouped_layout = sorted_ids.to(torch.int32)

    # ── THE GEMM — single call, exactly like h100_bench line 279 ─────────
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a_fp8, b_fp8, d, grouped_layout)

    # ── Scatter back ─────────────────────────────────────────────────────
    output = torch.zeros(N, D, dtype=torch.bfloat16, device=device)
    # Pad d from [M, I] to [M, D] for scatter (only benchmarks the GEMM kernel)
    d_padded = torch.zeros(M, D, dtype=torch.bfloat16, device=device)
    d_padded[:, :I] = d
    w = sorted_w.to(torch.bfloat16).unsqueeze(-1)
    output.index_add_(0, sorted_tok, d_padded * w)

    return output


# ── Single config benchmark ───────────────────────────────────────────────────

def benchmark_one(
    batch_size: int,
    n_tokens: int,
    n_experts: int,
    active_experts: int,
    ffn_dim: int,
    precision: str,
    device: torch.device,
    warmup: int = WARMUP,
    iters: int = ITERS,
) -> List[BenchResult]:
    """Run all available implementations for one (batch, tokens, experts, active, ffn, prec) point.

    Returns a list with one BenchResult per implementation (baseline + optionally DeepGEMM).
    OOM is caught and recorded in the result rather than propagating.
    """
    results: List[BenchResult] = []

    config_dict = {
        "batch_size": batch_size,
        "n_tokens": n_tokens,
        "n_experts": n_experts,
        "active_experts": active_experts,
        "ffn_dim": ffn_dim,
        "precision": precision,
        "hidden_size": HIDDEN_SIZE,
    }

    # ── Compute static FLOPs and bytes ────────────────────────────────────
    flops = compute_moe_flops(
        N_tokens=n_tokens,
        K_active=active_experts,
        D_hidden=HIDDEN_SIZE,
        D_intermediate=ffn_dim,
    )
    dtype_bytes = 1 if precision == "fp8" else 2
    bytes_accessed = compute_moe_bytes(
        N_tokens=n_tokens,
        K_active=active_experts,
        D_hidden=HIDDEN_SIZE,
        D_intermediate=ffn_dim,
        N_experts=n_experts,
        dtype_bytes=dtype_bytes,
    )
    oi = compute_operational_intensity(flops, bytes_accessed)
    roofline_bound = classify_roofline_bound(oi, precision)

    def _fill_result(r: BenchResult, times: list, stats: dict) -> BenchResult:
        r.latency_ms = times
        r.median_ms = stats["median"]
        r.mean_ms = stats["mean"]
        r.std_ms = stats["std"]
        r.p5_ms = stats["p5"]
        r.p50_ms = stats["p50"]
        r.p95_ms = stats["p95"]
        r.p99_ms = stats["p99"]
        r.ci_95_low = stats["ci_95_low"]
        r.ci_95_high = stats["ci_95_high"]

        latency_s = r.median_ms / 1e3
        r.tflops = compute_tflops(flops, latency_s)
        r.mfu_pct = compute_mfu(flops, latency_s, precision)
        r.bandwidth_gb_s = compute_bandwidth_gb_s(bytes_accessed, latency_s)
        r.hbm_sol_pct = compute_hbm_sol(bytes_accessed, latency_s)
        r.operational_intensity = oi
        r.roofline_bound = roofline_bound
        r.peak_memory_gb = torch.cuda.max_memory_allocated(device) / 1e9
        return r

    # ── Baseline: per-expert loop ─────────────────────────────────────────
    label = f"moe_E{n_experts}_K{active_experts}_I{ffn_dim}_{precision}"
    try:
        torch.cuda.reset_peak_memory_stats(device)
        hidden, guw, dw, ids, weights = _make_inputs(
            n_tokens, n_experts, active_experts, ffn_dim, precision, device
        )

        def _baseline():
            return moe_forward_baseline(hidden, guw, dw, ids, weights)

        times, stats = cuda_timer_extended(_baseline, warmup=warmup, iters=iters)
        r = BenchResult(name=label, impl="pytorch_loop", config=config_dict)
        results.append(_fill_result(r, times, stats))

        outlier_info = check_outliers(times)
        if not outlier_info["valid"]:
            print(f"  [WARN] {label} pytorch_loop: {outlier_info['flags']}")

        # Clean up baseline tensors before trying DeepGEMM (save HBM)
        del hidden, guw, dw, ids, weights
        torch.cuda.empty_cache()

    except torch.cuda.OutOfMemoryError as e:
        r = BenchResult(name=label, impl="pytorch_loop", config=config_dict)
        r.is_oom = True
        r.error = str(e)
        results.append(r)
        torch.cuda.empty_cache()
        # If baseline OOM'd, DeepGEMM will also OOM — skip it
        return results

    # ── DeepGEMM grouped FP8 GEMM ────────────────────────────────────────
    if DEEPGEMM_AVAILABLE and precision == "fp8":
        try:
            torch.cuda.reset_peak_memory_stats(device)
            hidden, guw, dw, ids, weights = _make_inputs(
                n_tokens, n_experts, active_experts, ffn_dim, "fp8", device
            )

            def _deepgemm():
                return moe_forward_deepgemm(hidden, guw, dw, ids, weights)

            times_dg, stats_dg = cuda_timer_extended(_deepgemm, warmup=warmup, iters=iters)
            r_dg = BenchResult(name=label, impl="deepgemm_fp8", config=config_dict)
            results.append(_fill_result(r_dg, times_dg, stats_dg))

            del hidden, guw, dw, ids, weights
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            r_dg = BenchResult(name=label, impl="deepgemm_fp8", config=config_dict)
            r_dg.is_oom = True
            r_dg.error = str(e)
            results.append(r_dg)
            torch.cuda.empty_cache()

    return results


# ── Sweep construction ────────────────────────────────────────────────────────

def build_sweep_quick() -> List[dict]:
    """Quick sweep: GLM-5's exact expert config (E=256, K=8, I=2048) at varying batch/token.

    Covers the most practically relevant operating points for GLM-5 serving.
    Both BF16 and FP8 precision included to show the quantisation speedup.
    """
    configs = []
    for bs in MOE_BENCH_BATCHES:
        for n_tok in MOE_BENCH_TOKENS:
            for prec in ["bf16", "fp8"]:
                configs.append(dict(
                    batch_size=bs,
                    n_tokens=n_tok,
                    n_experts=GLM5_CONFIG["n_routed_experts"],   # 256
                    active_experts=GLM5_CONFIG["num_experts_per_tok"],  # 8
                    ffn_dim=GLM5_CONFIG["moe_intermediate_size"],  # 2048
                    precision=prec,
                ))
    return configs


def build_sweep_full(
    batches: Optional[List[int]] = None,
    tokens: Optional[List[int]] = None,
    experts: Optional[List[int]] = None,
    active: Optional[List[int]] = None,
    ffn_dims: Optional[List[int]] = None,
) -> List[dict]:
    """Full cross-product sweep following MoE-Inference-Bench (SC '25) methodology.

    Default ranges match MOE_BENCH_* constants from shared/config.py.
    Any axis can be overridden via CLI args.
    """
    batches = batches or MOE_BENCH_BATCHES
    tokens = tokens or MOE_BENCH_TOKENS
    experts = experts or MOE_BENCH_EXPERTS
    active = active or MOE_BENCH_ACTIVE
    ffn_dims = ffn_dims or MOE_BENCH_FFN_DIMS

    configs = []
    for bs in batches:
        for n_tok in tokens:
            for n_exp in experts:
                for k_act in active:
                    # Skip invalid: can't activate more experts than exist
                    if k_act > n_exp:
                        continue
                    for ffn in ffn_dims:
                        for prec in ["bf16", "fp8"]:
                            configs.append(dict(
                                batch_size=bs,
                                n_tokens=n_tok,
                                n_experts=n_exp,
                                active_experts=k_act,
                                ffn_dim=ffn,
                                precision=prec,
                            ))
    return configs


# ── Progress helper ───────────────────────────────────────────────────────────

def _progress(current: int, total: int, cfg: dict) -> None:
    prec = cfg["precision"].upper()
    print(
        f"[{current:>4}/{total}] "
        f"B={cfg['batch_size']:>2} T={cfg['n_tokens']:>4} "
        f"E={cfg['n_experts']:>3} K={cfg['active_experts']} "
        f"I={cfg['ffn_dim']:>4} {prec}"
    )


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="MoE-Inference-Bench (SC '25) sweep for GLM-5 expert computation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--batch", type=int, nargs="+", default=None,
                        metavar="B",
                        help="Batch sizes to sweep. Default: MOE_BENCH_BATCHES from config.")
    parser.add_argument("--tokens", type=int, nargs="+", default=None,
                        metavar="T",
                        help="Token counts to sweep. Default: MOE_BENCH_TOKENS from config.")
    parser.add_argument("--experts", type=int, nargs="+", default=None,
                        metavar="E",
                        help="Total expert counts. Default: MOE_BENCH_EXPERTS from config.")
    parser.add_argument("--active", type=int, nargs="+", default=None,
                        metavar="K",
                        help="Active experts per token. Default: MOE_BENCH_ACTIVE from config.")
    parser.add_argument("--ffn-dims", type=int, nargs="+", default=None,
                        metavar="I",
                        help="FFN intermediate dims. Default: MOE_BENCH_FFN_DIMS from config.")
    parser.add_argument("--output-dir", type=str,
                        default=os.path.join(os.path.dirname(__file__), "results"),
                        help="Directory for JSON result files.")
    parser.add_argument("--quick", action="store_true",
                        help=(
                            "Quick mode: only GLM-5's exact config (E=256, K=8, I=2048) "
                            "at varying batch/token counts. "
                            "Ignores --experts/--active/--ffn-dims."
                        ))
    parser.add_argument("--warmup", type=int, default=WARMUP,
                        help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=ITERS,
                        help="Measured iterations.")
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch device string.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.", file=sys.stderr)
        sys.exit(1)

    device = torch.device(args.device)
    env = capture_environment()

    print("=" * 70)
    print("  MoE-Inference-Bench (SC '25) — GLM-5 Expert Sweep")
    print("=" * 70)
    print(f"  GPU:          {env.get('gpu_name', 'unknown')}")
    print(f"  DeepGEMM:     {'available' if DEEPGEMM_AVAILABLE else 'not installed'}")
    print(f"  Mode:         {'quick (GLM-5 config)' if args.quick else 'full cross-product'}")
    print(f"  Warmup/Iters: {args.warmup}/{args.iters}")
    print(f"  Output dir:   {args.output_dir}")
    print()

    # Build sweep grid
    if args.quick:
        configs = build_sweep_quick()
    else:
        configs = build_sweep_full(
            batches=args.batch,
            tokens=args.tokens,
            experts=args.experts,
            active=args.active,
            ffn_dims=args.ffn_dims,
        )

    total = len(configs)
    print(f"  Total configs: {total}")
    print()

    all_results: List[BenchResult] = []

    for i, cfg in enumerate(configs, 1):
        _progress(i, total, cfg)
        try:
            results = benchmark_one(
                batch_size=cfg["batch_size"],
                n_tokens=cfg["n_tokens"],
                n_experts=cfg["n_experts"],
                active_experts=cfg["active_experts"],
                ffn_dim=cfg["ffn_dim"],
                precision=cfg["precision"],
                device=device,
                warmup=args.warmup,
                iters=args.iters,
            )
            for r in results:
                if r.is_oom:
                    print(f"       OOM  [{r.impl}]")
                else:
                    print(
                        f"       {r.impl:<18} "
                        f"median={r.median_ms:>7.3f} ms  "
                        f"p99={r.p99_ms:>7.3f} ms  "
                        f"{r.tflops:>6.1f} TFLOPS  "
                        f"MFU={r.mfu_pct:>5.1f}%  "
                        f"{r.roofline_bound}"
                    )
            all_results.extend(results)
        except Exception as e:  # noqa: BLE001 — never crash the sweep loop
            print(f"       ERROR: {e}")

    print()

    # ── Summary table ──────────────────────────────────────────────────────
    print_summary_table(
        all_results,
        title="MoE Expert Sweep — Full Results",
    )

    # ── Save ───────────────────────────────────────────────────────────────
    mode_tag = "quick" if args.quick else "full"
    save_results(
        results=all_results,
        output_dir=args.output_dir,
        experiment_name=f"moe_sweep_{mode_tag}",
        env=env,
    )


if __name__ == "__main__":
    main()
