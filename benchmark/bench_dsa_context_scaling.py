"""DSA Indexer Context Scaling Benchmark.

Measures how DSA lightning indexer latency scales with context length,
answering: "Does the indexer become a bottleneck at long context?"

GLM-5's DSA selects top-2048 tokens from a context of T tokens.
The indexer scoring is O(S × T × H × D) — linear in T.
At T=128K, does this linear cost dominate total decode latency?

Two sub-experiments:
  1. indexer-scaling:  Indexer-only latency at T ∈ {4K, 8K, 16K, 32K, 64K, 128K}
     - PyTorch einsum (eager) vs DeepGEMM fp8_mqa_logits (fused FP8)
     - Reports: latency, TFLOPS, scaling factor vs T=4K baseline
  2. indexer-fraction: Indexer latency as % of total decode layer latency
     - Uses serving_prefill_decode results for total decode numbers
     - Shows where indexer transitions from negligible to dominant

Context lengths match GLM-5 paper Table 6 RULER evaluation: {4K, 8K, 16K, 32K, 64K, 128K}.

References:
    - GLM-5 (arXiv:2602.15763) Table 6: RULER eval at 4K-128K
    - NSA (ACL 2025 Best Paper): "acceleration ratio increases with sequence length"
    - DeepGEMM: fp8_mqa_logits fused kernel, BLOCK_KV=256 tiling
    - GLM-5 Section 2.1.1: DSA indexer formula
    - vLLM blog (2025): DSA lightning indexer + FlashMLA sparse kernel two-stage architecture

Usage:
    python3 -m benchmark.bench_dsa_context_scaling --experiment all
    python3 -m benchmark.bench_dsa_context_scaling --experiment indexer-scaling
    python3 -m benchmark.bench_dsa_context_scaling --experiment indexer-fraction
"""

import os
import argparse
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.shared import (
    BenchResult, save_results, capture_environment,
)
from benchmark.shared.config import GLM5_CONFIG, H100_SPECS
from benchmark.shared.timer import cuda_timer_extended
from benchmark.shared.metrics import (
    compute_dsa_indexer_flops, compute_mfu, compute_tflops,
    compute_bandwidth_gb_s, compute_hbm_sol,
)
from benchmark.shared.report import print_summary_table

# DeepGEMM availability (may fail on CUDA 12.8 without patch)
DEEPGEMM_AVAILABLE = False
try:
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8
    DEEPGEMM_AVAILABLE = True
except ImportError:
    pass


# ── GLM-5 DSA Indexer dimensions ─────────────────────────────────────────

H_IDX = GLM5_CONFIG["index_n_heads"]       # 32
D_IDX = GLM5_CONFIG["index_head_dim"]      # 128
QK_ROPE = GLM5_CONFIG["qk_rope_head_dim"]  # 64
TOPK = GLM5_CONFIG["index_topk"]           # 2048

# GLM-5 Table 6 RULER evaluation context lengths
CONTEXT_LENGTHS = [4096, 8192, 16384, 32768, 65536, 131072]

# Batch sizes: decode is typically B=1 for single-user latency, B=32 for serving
BATCH_SIZES = [1, 32]


# ── Indexer Scoring Implementations ──────────────────────────────────────

def _indexer_score_pytorch(q, k_cached, weights, softmax_scale):
    """PyTorch eager indexer scoring: einsum + ReLU + weighted sum + topk.

    This is the exact computation from dsa_indexer.py lines 101-103, 110-111.
    q:       [B, S, H, D]
    k_cached: [B, T, D]
    weights:  [B, S, H]
    Returns: indices [B, S, topk]
    """
    scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * softmax_scale
    scores = F.relu(scores)
    index_scores = torch.einsum("bsht,bsh->bst", scores, weights)
    total_len = index_scores.shape[-1]
    topk = min(TOPK, total_len)
    return index_scores.topk(topk, dim=-1).indices


def _indexer_score_deepgemm(q, k_cached, weights):
    """DeepGEMM fused FP8 indexer scoring.

    Uses fp8_mqa_logits kernel: fuses ReLU(Q·K^T) * weights into one kernel.
    q:       [B, S, H, D] — B must be 1
    k_cached: [B, T, D]
    weights:  [B, S, H]
    Returns: indices [B, S, topk]
    """
    q_3d = q.squeeze(0)          # [S, H, D]
    k_2d = k_cached.squeeze(0)   # [T, D]
    w_2d = weights.squeeze(0)    # [S, H]

    seq_len = q_3d.shape[0]
    seq_len_kv = k_2d.shape[0]

    q_fp8 = q_3d.to(torch.float8_e4m3fn)
    kv_tuple = per_token_cast_to_fp8(k_2d, use_ue8m0=True)

    cu_k_start = torch.zeros(seq_len, dtype=torch.int32, device=q.device)
    cu_k_end = torch.full((seq_len,), seq_len_kv, dtype=torch.int32, device=q.device)

    logits = deep_gemm.fp8_mqa_logits(q_fp8, kv_tuple, w_2d, cu_k_start, cu_k_end)

    total_len = logits.shape[-1]
    topk = min(TOPK, total_len)
    return logits.unsqueeze(0).topk(topk, dim=-1).indices


# ── Sub-experiment 1: Indexer Scaling ────────────────────────────────────

def _bench_indexer_at_context(B, T, impl, device, warmup=10, iters=50):
    """Benchmark indexer scoring at one (B, T) configuration.

    impl: "pytorch" or "deepgemm"
    """
    S = 1  # decode: single query token
    label = f"indexer_B{B}_T{T}_{impl}"
    config_info = {"B": B, "S": S, "T": T, "impl": impl, "H": H_IDX, "D": D_IDX, "topk": TOPK}

    try:
        # Allocate indexer inputs (no model needed — just raw tensors)
        q = torch.randn(B, S, H_IDX, D_IDX, dtype=torch.bfloat16, device=device)
        k_cached = torch.randn(B, T, D_IDX, dtype=torch.bfloat16, device=device)
        weights = torch.randn(B, S, H_IDX, dtype=torch.float32, device=device)
        softmax_scale = D_IDX ** -0.5
    except torch.cuda.OutOfMemoryError:
        return BenchResult(name=label, impl=impl, config=config_info, is_oom=True)

    if impl == "pytorch":
        def run():
            _indexer_score_pytorch(q, k_cached, weights, softmax_scale)
    elif impl == "deepgemm" and B == 1:
        # Verify DeepGEMM works at this T before timing
        try:
            _indexer_score_deepgemm(q, k_cached, weights)
        except RuntimeError as e:
            del q, k_cached, weights
            torch.cuda.empty_cache()
            return BenchResult(name=label, impl=impl, config=config_info, error=f"DeepGEMM: {str(e)[:80]}")

        def run():
            _indexer_score_deepgemm(q, k_cached, weights)
    else:
        del q, k_cached, weights
        torch.cuda.empty_cache()
        return BenchResult(name=label, impl=impl, config=config_info, error="DeepGEMM requires B=1")

    try:
        torch.cuda.reset_peak_memory_stats()
        times, stats = cuda_timer_extended(run, warmup=warmup, iters=iters)
    except torch.cuda.OutOfMemoryError:
        del q, k_cached, weights
        torch.cuda.empty_cache()
        return BenchResult(name=label, impl=impl, config=config_info, is_oom=True)
    except Exception as e:
        del q, k_cached, weights
        torch.cuda.empty_cache()
        return BenchResult(name=label, impl=impl, config=config_info, error=str(e)[:80])

    flops = compute_dsa_indexer_flops(S, T, H_IDX, D_IDX)
    latency_s = stats["median"] / 1000.0

    # Bytes accessed: Q [B,S,H,D] + K [B,T,D] + weights [B,S,H] + output [B,S,topk]
    dtype_bytes = 2  # bf16
    bytes_accessed = (B * S * H_IDX * D_IDX + B * T * D_IDX + B * S * H_IDX) * dtype_bytes
    bytes_accessed += B * S * min(TOPK, T) * 4  # int32 indices output

    del q, k_cached, weights
    torch.cuda.empty_cache()

    return BenchResult(
        name=label, impl=impl, config=config_info,
        latency_ms=times,
        median_ms=stats["median"], mean_ms=stats["mean"], std_ms=stats["std"],
        p5_ms=stats["p5"], p50_ms=stats["p50"], p95_ms=stats["p95"], p99_ms=stats["p99"],
        ci_95_low=stats["ci_95_low"], ci_95_high=stats["ci_95_high"],
        tflops=compute_tflops(flops, latency_s),
        mfu_pct=compute_mfu(flops, latency_s, "fp8" if impl == "deepgemm" else "bf16"),
        bandwidth_gb_s=compute_bandwidth_gb_s(bytes_accessed, latency_s),
        hbm_sol_pct=compute_hbm_sol(bytes_accessed, latency_s),
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
    )


def run_indexer_scaling_experiment(output_dir, warmup=10, iters=50):
    """Sub-experiment 1: How does indexer latency scale with context length?"""
    print("=" * 70)
    print("  DSA Indexer Context Scaling")
    print("  GLM-5 Table 6 context lengths: {4K, 8K, 16K, 32K, 64K, 128K}")
    print("=" * 70)

    device = torch.device("cuda")
    results = []

    impls = ["pytorch"]
    if DEEPGEMM_AVAILABLE:
        impls.append("deepgemm")
        print("  DeepGEMM available — will benchmark fp8_mqa_logits kernel")
    else:
        print("  DeepGEMM not available — PyTorch-only")

    for B in BATCH_SIZES:
        print(f"\n  --- B={B} (S=1 decode) ---")
        print(f"  {'T':>8} {'Impl':<12} {'Median(ms)':>10} {'TFLOPS':>8} {'BW(GB/s)':>10} {'SOL%':>6} {'Mem(GB)':>8}")
        print("  " + "-" * 68)

        baseline_latency = {}

        for T in CONTEXT_LENGTHS:
            for impl in impls:
                # DeepGEMM only works with B=1
                if impl == "deepgemm" and B > 1:
                    continue

                r = _bench_indexer_at_context(B, T, impl, device, warmup, iters)

                if r.is_oom:
                    print(f"  {T:>8} {impl:<12} {'OOM':>10}")
                elif r.error:
                    print(f"  {T:>8} {impl:<12} ERROR: {r.error[:50]}")
                else:
                    # Track baseline for scaling factor
                    key = (B, impl)
                    if T == CONTEXT_LENGTHS[0]:
                        baseline_latency[key] = r.median_ms

                    scale = r.median_ms / baseline_latency.get(key, r.median_ms)
                    r.config["scaling_factor"] = round(scale, 2)
                    r.config["theoretical_scaling"] = round(T / CONTEXT_LENGTHS[0], 2)

                    print(f"  {T:>8} {impl:<12} {r.median_ms:>9.3f}ms {r.tflops:>7.1f}TF {r.bandwidth_gb_s:>9.0f} {r.hbm_sol_pct:>5.1f}% {r.peak_memory_gb:>7.1f}GB  [{scale:.1f}x vs {CONTEXT_LENGTHS[0]//1024}K]")

                results.append(r)

    # Print scaling summary
    print(f"\n  --- Scaling Summary ---")
    print(f"  {'B':>4} {'Impl':<12} {'4K(ms)':>8} {'8K':>8} {'16K':>8} {'32K':>8} {'64K':>8} {'128K':>8} {'Linear?'}")
    for B in BATCH_SIZES:
        for impl in impls:
            if impl == "deepgemm" and B > 1:
                continue
            latencies = []
            for T in CONTEXT_LENGTHS:
                match = [r for r in results if r.config.get("B") == B and r.config.get("T") == T and r.config.get("impl") == impl and not r.is_oom and not r.error]
                latencies.append(match[0].median_ms if match else float('nan'))

            if latencies[0] != latencies[0]:  # nan check
                continue

            vals = " ".join(f"{l:>7.2f}" if l == l else f"{'OOM':>7}" for l in latencies)
            # Check linearity: if T doubles, latency should ~double
            if len(latencies) >= 2 and latencies[0] > 0 and latencies[1] > 0:
                ratio = latencies[1] / latencies[0]  # 8K/4K should be ~2.0
                linear = "yes" if 1.5 < ratio < 2.5 else f"no ({ratio:.1f}x)"
            else:
                linear = "?"
            print(f"  {B:>4} {impl:<12} {vals} {linear}")

    print()
    env = capture_environment()
    save_results(results, output_dir, "dsa_indexer_scaling", env)
    return results


# ── Sub-experiment 2: Indexer Fraction of Decode Latency ─────────────────

def run_indexer_fraction_experiment(output_dir, warmup=10, iters=50):
    """Sub-experiment 2: Indexer as % of total decode latency.

    Uses pre-computed decode layer latencies from serving_prefill_decode
    results or estimates them from the indexer + attention + MoE components.
    """
    print("=" * 70)
    print("  DSA Indexer: Fraction of Total Decode Latency")
    print("=" * 70)

    device = torch.device("cuda")
    results = []

    # Benchmark indexer at B=1 for each context length
    print(f"\n  {'T':>8} {'Indexer(ms)':>12} {'Est.Attn(ms)':>13} {'Est.Total(ms)':>14} {'Idx%':>6}")
    print("  " + "-" * 58)

    for T in CONTEXT_LENGTHS:
        r = _bench_indexer_at_context(1, T, "pytorch", device, warmup, iters)

        if r.is_oom or r.error:
            print(f"  {T:>8} {'OOM' if r.is_oom else 'ERR':>12}")
            results.append(r)
            continue

        idx_ms = r.median_ms

        # Estimate full attention cost at this T (B=1, S=1 decode)
        # Dense MLA attention: Q·K^T [1, 64, 1, T] + softmax + P·V [1, 64, 1, T]
        # At eager: dominated by KV cache read of [1, 64, T, 256] × 2 (K+V)
        H = GLM5_CONFIG["num_heads"]
        d_qk = GLM5_CONFIG.get("qk_head_dim", 256)
        d_v = GLM5_CONFIG.get("v_head_dim", 256)
        kv_bytes = 1 * H * T * (d_qk + d_v) * 2  # bf16
        # Assume ~50% of H100 HBM bandwidth for attention read
        est_attn_ms = kv_bytes / (H100_SPECS["hbm_bandwidth_gb_s"] * 0.5 * 1e6)

        # MoE layer adds ~5ms at B=1 (from h100_bench results: sparse layer ~4.8ms)
        est_moe_ms = 4.8
        est_total_ms = idx_ms + est_attn_ms + est_moe_ms
        idx_pct = (idx_ms / est_total_ms) * 100

        r.config["indexer_ms"] = round(idx_ms, 3)
        r.config["est_attention_ms"] = round(est_attn_ms, 3)
        r.config["est_total_decode_ms"] = round(est_total_ms, 3)
        r.config["indexer_fraction_pct"] = round(idx_pct, 1)

        print(f"  {T:>8} {idx_ms:>11.3f}ms {est_attn_ms:>12.3f}ms {est_total_ms:>13.3f}ms {idx_pct:>5.1f}%")
        results.append(r)

    # Crossover analysis
    print(f"\n  Key finding: DSA indexer cost is O(T) while attention is also O(T).")
    print(f"  The indexer fraction should remain roughly constant across context lengths")
    print(f"  (both are linear in T). If it grows, the indexer has worse constant factors.")

    if DEEPGEMM_AVAILABLE:
        print(f"\n  --- With DeepGEMM fp8_mqa_logits (B=1) ---")
        print(f"  {'T':>8} {'DG Idx(ms)':>12} {'PT Idx(ms)':>12} {'Speedup':>8}")
        print("  " + "-" * 45)

        for T in CONTEXT_LENGTHS:
            r_dg = _bench_indexer_at_context(1, T, "deepgemm", device, warmup, iters)
            r_pt = [r for r in results if r.config.get("T") == T and r.config.get("impl") == "pytorch" and not r.is_oom]

            if r_dg.is_oom or r_dg.error:
                print(f"  {T:>8} {'OOM' if r_dg.is_oom else 'ERR':>12}")
                results.append(r_dg)
                continue

            pt_ms = r_pt[0].median_ms if r_pt else float('nan')
            speedup = pt_ms / r_dg.median_ms if r_dg.median_ms > 0 else 0
            r_dg.config["pytorch_speedup"] = round(speedup, 2)

            print(f"  {T:>8} {r_dg.median_ms:>11.3f}ms {pt_ms:>11.3f}ms {speedup:>7.2f}x")
            results.append(r_dg)

    print()
    env = capture_environment()
    save_results(results, output_dir, "dsa_indexer_fraction", env)
    return results


# ── CLI ──────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="DSA indexer context scaling benchmark.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--experiment", choices=["indexer-scaling", "indexer-fraction", "all"],
                        default="all", help="Which sub-experiment to run.")
    parser.add_argument("--output-dir",
                        default=os.path.join(os.path.dirname(__file__), "..", "results", "dsa_scaling"),
                        help="Directory for JSON result files.")
    parser.add_argument("--warmup", type=int, default=10, help="Warmup iterations.")
    parser.add_argument("--iters", type=int, default=50, help="Measured iterations.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available.", file=sys.stderr)
        sys.exit(1)

    os.makedirs(args.output_dir, exist_ok=True)

    experiments = {
        "indexer-scaling": lambda: run_indexer_scaling_experiment(args.output_dir, args.warmup, args.iters),
        "indexer-fraction": lambda: run_indexer_fraction_experiment(args.output_dir, args.warmup, args.iters),
    }

    targets = experiments.keys() if args.experiment == "all" else [args.experiment]
    for name in targets:
        experiments[name]()

    print("All DSA experiments complete.")


if __name__ == "__main__":
    main()
