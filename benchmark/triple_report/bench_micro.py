"""Kernel-level microbenchmarks for the 12 GLM-5 components.

For each component, benchmarks are run at full GLM-5 dims:
  - Prefill:  B=1,  S=128
  - Decode:   B=32, T=4096  (full KV cache context)

Each component reports, per implementation (PyTorch / Triton / CUDA kernel):
  - median_ms, p99_ms
  - TFLOPS, MFU%, HBM SOL%, roofline bound

A side-by-side summary table is printed at the end, following the
"triple report" format of the GLM-5 arXiv paper.

Components benchmarked:
  1.  rmsnorm        — Triton fast_rms_layernorm vs PyTorch manual
  2.  swiglu         — Triton swiglu_fg_kernel vs F.silu()*gate
  3.  cross_entropy  — Triton chunked CE vs F.cross_entropy (vocab=154880)
  4.  rope           — PyTorch rotate_half (64 rope dims)
  5.  moe_router     — sigmoid + topk (n_group=1, flat routing)
  6.  dsa_indexer    — einsum scoring path (+ DeepGEMM if available)
  7.  dsa_sparse_attn— masked matmul (+ FlashMLA sparse if available)
  8.  mla_attention  — eager attention (+ FlashMLA if available)
  9.  moe_forward    — per-expert loop (+ DeepGEMM grouped FP8 if available)

References:
  - GLM-5 (arXiv:2602.15763)
  - FlashAttention-3 (Dao, 2024): 740 TFLOPS FP16 on H100
  - FlashMLA (2025): 660 TFLOPS dense / 410 TFLOPS sparse decode
  - DeepGEMM (2025): 1550 TFLOPS FP8 grouped GEMM
"""

import argparse
import sys
import os
import math
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F

# ── Shared utilities ─────────────────────────────────────────────────────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from benchmark.shared.config import (
    BenchResult,
    GLM5_CONFIG,
    H100_SPECS,
)
from benchmark.shared.timer import cuda_timer_extended, check_outliers
from benchmark.shared.metrics import (
    compute_moe_flops,
    compute_moe_bytes,
    compute_attention_flops,
    compute_attention_bytes,
    compute_dsa_indexer_flops,
    compute_mfu,
    compute_hbm_sol,
    compute_tflops,
    compute_bandwidth_gb_s,
    compute_operational_intensity,
    classify_roofline_bound,
)
from benchmark.shared.report import save_results, print_summary_table, capture_environment


# ── Optional kernel imports ───────────────────────────────────────────────────
# All wrapped in try/except — the script runs the PyTorch baseline regardless.

try:
    import triton  # noqa: F401 — checked for availability
    from triton_kernels import fast_rms_layernorm  # project-local Triton kernel
    TRITON_RMSNORM_AVAILABLE = True
except ImportError:
    TRITON_RMSNORM_AVAILABLE = False

try:
    from triton_kernels import swiglu_fg_kernel  # project-local Triton SwiGLU
    TRITON_SWIGLU_AVAILABLE = True
except ImportError:
    TRITON_SWIGLU_AVAILABLE = False

try:
    from triton_kernels import chunked_cross_entropy  # project-local chunked CE
    TRITON_CE_AVAILABLE = True
except ImportError:
    TRITON_CE_AVAILABLE = False

try:
    import flash_mla
    FLASHMLA_AVAILABLE = True
    print("[bench_micro] FlashMLA available — MLA + DSA sparse paths enabled.")
except ImportError:
    FLASHMLA_AVAILABLE = False
    print("[bench_micro] FlashMLA not found — falling back to PyTorch eager attention.")

try:
    import deep_gemm
    DEEPGEMM_AVAILABLE = True
    print("[bench_micro] DeepGEMM available — FP8 grouped GEMM path enabled.")
except ImportError:
    DEEPGEMM_AVAILABLE = False
    print("[bench_micro] DeepGEMM not found — MoE forward uses PyTorch loop only.")


# ── GLM-5 dimension aliases ───────────────────────────────────────────────────
D         = GLM5_CONFIG["hidden_size"]            # 6144
N_HEADS   = GLM5_CONFIG["num_heads"]              # 64
KV_LORA   = GLM5_CONFIG["kv_lora_rank"]          # 512
QK_ROPE   = GLM5_CONFIG["qk_rope_head_dim"]      # 64   (RoPE dims per head)
QK_NOPE  = GLM5_CONFIG["qk_nope_head_dim"]       # 192
QK_HEAD   = GLM5_CONFIG["qk_head_dim"]            # 256  = 192 + 64
V_HEAD    = GLM5_CONFIG["v_head_dim"]             # 256
D_QK_ABS  = GLM5_CONFIG["d_qk_absorbed"]          # 576
D_V_ABS   = GLM5_CONFIG["d_v_absorbed"]           # 512
Q_LORA    = GLM5_CONFIG["q_lora_rank"]            # 2048
N_EXP     = GLM5_CONFIG["n_routed_experts"]       # 256
K_ACT     = GLM5_CONFIG["num_experts_per_tok"]    # 8
FFN_DIM   = GLM5_CONFIG["moe_intermediate_size"]  # 2048
VOCAB     = GLM5_CONFIG["vocab_size"]             # 154880
IDX_H     = GLM5_CONFIG["index_n_heads"]          # 32
IDX_D     = GLM5_CONFIG["index_head_dim"]         # 128
IDX_TOPK  = GLM5_CONFIG["index_topk"]             # 2048  (selected KV positions)
DENSE_FFN = GLM5_CONFIG["intermediate_size"]      # 12288  (dense layer FFN width)

# Benchmark operating points (GLM-5 serving reality)
PREFILL_B = 1
PREFILL_S = 128
DECODE_B  = 32
DECODE_T  = 4096   # full KV context length

WARMUP = 10
ITERS  = 100


# ── Utility: populate a BenchResult from timer output ───────────────────────

def _fill(
    r: BenchResult,
    times: List[float],
    stats: Dict,
    flops: int,
    bytes_accessed: int,
    precision: str = "bf16",
) -> BenchResult:
    r.latency_ms = times
    r.median_ms  = stats["median"]
    r.mean_ms    = stats["mean"]
    r.std_ms     = stats["std"]
    r.p5_ms      = stats["p5"]
    r.p50_ms     = stats["p50"]
    r.p95_ms     = stats["p95"]
    r.p99_ms     = stats["p99"]
    r.ci_95_low  = stats["ci_95_low"]
    r.ci_95_high = stats["ci_95_high"]

    lat_s = r.median_ms / 1e3
    r.tflops           = compute_tflops(flops, lat_s)
    r.mfu_pct          = compute_mfu(flops, lat_s, precision)
    r.bandwidth_gb_s   = compute_bandwidth_gb_s(bytes_accessed, lat_s)
    r.hbm_sol_pct      = compute_hbm_sol(bytes_accessed, lat_s)
    r.operational_intensity = compute_operational_intensity(flops, bytes_accessed)
    r.roofline_bound   = classify_roofline_bound(r.operational_intensity, precision)
    r.peak_memory_gb   = torch.cuda.max_memory_allocated() / 1e9
    return r


def _oom_result(name: str, impl: str, cfg: dict, exc: Exception) -> BenchResult:
    r = BenchResult(name=name, impl=impl, config=cfg)
    r.is_oom = True
    r.error  = str(exc)
    return r


# ═══════════════════════════════════════════════════════════════════════════════
# 1. RMSNorm
# ═══════════════════════════════════════════════════════════════════════════════

def bench_rmsnorm(device: torch.device) -> List[BenchResult]:
    """RMSNorm at full GLM-5 hidden dim (D=6144).

    FLOPs: 2 * N * D  (square + sum for RMS, then scale)
    Bytes: 3 * N * D * 2  (read x, read weight, write y — BF16)
    """
    results = []

    # Both prefill and decode shapes
    for mode, B, S in [("prefill", PREFILL_B, PREFILL_S), ("decode", DECODE_B, DECODE_T)]:
        N = B * S
        cfg = {"mode": mode, "B": B, "S": S, "D": D}
        name = f"rmsnorm_{mode}"

        flops        = 2 * N * D          # RMS + scale
        bytes_acc    = 3 * N * D * 2      # x (r), weight (r), y (w)  — BF16=2B

        x      = torch.randn(N, D, dtype=torch.bfloat16, device=device)
        weight = torch.ones(D, dtype=torch.bfloat16, device=device)
        eps    = 1e-5

        # ── PyTorch manual ────────────────────────────────────────────────
        def _pt_rmsnorm():
            rms = x.float().pow(2).mean(-1, keepdim=True).add(eps).sqrt()
            return (x / rms) * weight

        torch.cuda.reset_peak_memory_stats(device)
        times, stats = cuda_timer_extended(_pt_rmsnorm, WARMUP, ITERS)
        r = BenchResult(name=name, impl="pytorch_manual", config=cfg)
        results.append(_fill(r, times, stats, flops, bytes_acc))

        # ── Triton fast_rms_layernorm ─────────────────────────────────────
        if TRITON_RMSNORM_AVAILABLE:
            try:
                def _triton_rmsnorm():
                    return fast_rms_layernorm(x, weight, eps)

                torch.cuda.reset_peak_memory_stats(device)
                times_t, stats_t = cuda_timer_extended(_triton_rmsnorm, WARMUP, ITERS)
                r_t = BenchResult(name=name, impl="triton_fast_rms", config=cfg)
                results.append(_fill(r_t, times_t, stats_t, flops, bytes_acc))
            except torch.cuda.OutOfMemoryError as e:
                results.append(_oom_result(name, "triton_fast_rms", cfg, e))
                torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 2. SwiGLU
# ═══════════════════════════════════════════════════════════════════════════════

def bench_swiglu(device: torch.device) -> List[BenchResult]:
    """SwiGLU activation on the dense FFN intermediate (I=12288).

    FLOPs: 2 * N * I  (SiLU gate + elementwise multiply)
    Bytes: 3 * N * I * 2  (gate r, up r, out w)
    """
    results = []

    for mode, B, S in [("prefill", PREFILL_B, PREFILL_S), ("decode", DECODE_B, DECODE_T)]:
        N = B * S
        I = DENSE_FFN
        cfg  = {"mode": mode, "B": B, "S": S, "I": I}
        name = f"swiglu_{mode}"

        flops     = 2 * N * I        # silu(gate)*up
        bytes_acc = 3 * N * I * 2    # gate+up read, out write

        gate = torch.randn(N, I, dtype=torch.bfloat16, device=device)
        up   = torch.randn(N, I, dtype=torch.bfloat16, device=device)

        # ── PyTorch baseline ──────────────────────────────────────────────
        def _pt_swiglu():
            return F.silu(gate) * up

        torch.cuda.reset_peak_memory_stats(device)
        times, stats = cuda_timer_extended(_pt_swiglu, WARMUP, ITERS)
        r = BenchResult(name=name, impl="pytorch_silu_mul", config=cfg)
        results.append(_fill(r, times, stats, flops, bytes_acc))

        # ── Triton fused SwiGLU kernel ────────────────────────────────────
        if TRITON_SWIGLU_AVAILABLE:
            try:
                # swiglu_fg_kernel accepts fused [N, 2*I] — split internally
                fused = torch.cat([gate, up], dim=-1)

                def _triton_swiglu():
                    return swiglu_fg_kernel(fused)

                torch.cuda.reset_peak_memory_stats(device)
                times_t, stats_t = cuda_timer_extended(_triton_swiglu, WARMUP, ITERS)
                r_t = BenchResult(name=name, impl="triton_swiglu_fused", config=cfg)
                results.append(_fill(r_t, times_t, stats_t, flops, bytes_acc))
            except torch.cuda.OutOfMemoryError as e:
                results.append(_oom_result(name, "triton_swiglu_fused", cfg, e))
                torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Cross-Entropy (vocab=154880)
# ═══════════════════════════════════════════════════════════════════════════════

def bench_cross_entropy(device: torch.device) -> List[BenchResult]:
    """Cross-entropy at GLM-5 vocab size (154880).

    FLOPs: ~2 * N * V  (logit max-subtract + exp + log — softmax + NLL)
    Bytes: N * V * 2  (read logits BF16) + N * 8 (read labels int64)
    """
    results = []

    # CE is only relevant at prefill/train time
    B, S = PREFILL_B, PREFILL_S
    N = B * S
    V = VOCAB
    cfg  = {"mode": "prefill", "B": B, "S": S, "V": V}
    name = "cross_entropy"

    flops     = 2 * N * V
    bytes_acc = N * V * 2 + N * 8   # logits (BF16) + labels (int64)

    logits = torch.randn(N, V, dtype=torch.bfloat16, device=device)
    labels = torch.randint(0, V, (N,), device=device)

    # ── PyTorch F.cross_entropy ───────────────────────────────────────────
    def _pt_ce():
        return F.cross_entropy(logits.float(), labels)

    torch.cuda.reset_peak_memory_stats(device)
    times, stats = cuda_timer_extended(_pt_ce, WARMUP, ITERS)
    r = BenchResult(name=name, impl="pytorch_cross_entropy", config=cfg)
    results.append(_fill(r, times, stats, flops, bytes_acc))

    # ── Triton chunked cross-entropy ──────────────────────────────────────
    if TRITON_CE_AVAILABLE:
        try:
            def _triton_ce():
                return chunked_cross_entropy(logits, labels)

            torch.cuda.reset_peak_memory_stats(device)
            times_t, stats_t = cuda_timer_extended(_triton_ce, WARMUP, ITERS)
            r_t = BenchResult(name=name, impl="triton_chunked_ce", config=cfg)
            results.append(_fill(r_t, times_t, stats_t, flops, bytes_acc))
        except torch.cuda.OutOfMemoryError as e:
            results.append(_oom_result(name, "triton_chunked_ce", cfg, e))
            torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 4. RoPE
# ═══════════════════════════════════════════════════════════════════════════════

def bench_rope(device: torch.device) -> List[BenchResult]:
    """Rotary position embedding (RoPE) with QK_ROPE=64 head dims.

    FLOPs: 6 * N * H * D_rope  (rotate_half: 2 splits, 2 negates, 2 adds/muls)
    Bytes: 3 * N * H * D_rope * 2  (q/k read/write, cos/sin read)
    """
    results = []

    for mode, B, S in [("prefill", PREFILL_B, PREFILL_S), ("decode", DECODE_B, DECODE_T)]:
        N = B * S
        H = N_HEADS     # 64 heads
        R = QK_ROPE     # 64 rope dims per head
        cfg  = {"mode": mode, "B": B, "S": S, "H": H, "D_rope": R}
        name = f"rope_{mode}"

        flops     = 6 * N * H * R
        bytes_acc = (2 * N * H * R + N * H * R) * 2   # q+k (r+w) + cos/sin (r)

        q   = torch.randn(B, S, H, R, dtype=torch.bfloat16, device=device)
        k   = torch.randn(B, S, H, R, dtype=torch.bfloat16, device=device)
        cos = torch.randn(B, S, 1, R, dtype=torch.bfloat16, device=device)
        sin = torch.randn(B, S, 1, R, dtype=torch.bfloat16, device=device)

        def _rotate_half(x: torch.Tensor) -> torch.Tensor:
            half = x.shape[-1] // 2
            x1, x2 = x[..., :half], x[..., half:]
            return torch.cat([-x2, x1], dim=-1)

        def _pt_rope():
            q_rot = q * cos + _rotate_half(q) * sin
            k_rot = k * cos + _rotate_half(k) * sin
            return q_rot, k_rot

        torch.cuda.reset_peak_memory_stats(device)
        times, stats = cuda_timer_extended(_pt_rope, WARMUP, ITERS)
        r = BenchResult(name=name, impl="pytorch_rotate_half", config=cfg)
        results.append(_fill(r, times, stats, flops, bytes_acc))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 5. MoE Router
# ═══════════════════════════════════════════════════════════════════════════════

def bench_moe_router(device: torch.device) -> List[BenchResult]:
    """Sigmoid + TopK routing (GLM-5 flat routing, n_group=1).

    FLOPs: N * E (sigmoid) + N * E * log2(K) (topk approx)
    Bytes: 2 * N * E * 4  (router_logits read + sigmoid output)
    """
    results = []

    for mode, B, S in [("prefill", PREFILL_B, PREFILL_S), ("decode", DECODE_B, DECODE_T)]:
        N = B * S
        E = N_EXP   # 256
        K = K_ACT   # 8
        cfg  = {"mode": mode, "B": B, "S": S, "E": E, "K": K, "n_group": 1}
        name = f"moe_router_{mode}"

        flops     = N * E + N * E  # sigmoid + topk (approx same cost as sigmoid at this scale)
        bytes_acc = N * E * 4 * 2  # logits read (fp32) + scores write

        router_logits = torch.randn(N, E, dtype=torch.float32, device=device)

        def _pt_router():
            scores = torch.sigmoid(router_logits)                # [N, E]
            topk_w, topk_ids = torch.topk(scores, K, dim=-1)    # [N, K]
            # L1 normalise routing weights
            topk_w = topk_w / topk_w.sum(dim=-1, keepdim=True).clamp(min=1e-6)
            return topk_w, topk_ids

        torch.cuda.reset_peak_memory_stats(device)
        times, stats = cuda_timer_extended(_pt_router, WARMUP, ITERS)
        r = BenchResult(name=name, impl="pytorch_sigmoid_topk", config=cfg)
        results.append(_fill(r, times, stats, flops, bytes_acc))

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 6. DSA Indexer
# ═══════════════════════════════════════════════════════════════════════════════

def bench_dsa_indexer(device: torch.device) -> List[BenchResult]:
    """DSA lightning indexer: score each (query, key) pair via multi-head dot products.

    Uses the einsum path: score[s,t] = relu(einsum('shd,thd->st', q_idx, k_idx) dot w_h)
    Full GLM-5 shape: H_idx=32 heads, D_idx=128, S_q × S_kv attention budget.

    FLOPs from compute_dsa_indexer_flops.
    Bytes: q [S_q, H_idx, D_idx] + k [S_kv, H_idx, D_idx] + w [H_idx]
    """
    results = []

    for mode, B, S, T in [
        ("prefill", PREFILL_B, PREFILL_S, PREFILL_S),
        ("decode",  DECODE_B,  1,         DECODE_T),   # single-step decode
    ]:
        S_q  = S
        S_kv = T
        H    = IDX_H    # 32
        Dh   = IDX_D    # 128
        cfg  = {"mode": mode, "B": B, "S_q": S_q, "S_kv": S_kv, "H_idx": H, "D_idx": Dh}
        name = f"dsa_indexer_{mode}"

        flops     = compute_dsa_indexer_flops(S_q, S_kv, H, Dh)
        bytes_acc = (S_q * H * Dh + S_kv * H * Dh + H + S_q * S_kv) * 2  # BF16

        q_idx = torch.randn(B, S_q,  H, Dh, dtype=torch.bfloat16, device=device)
        k_idx = torch.randn(B, S_kv, H, Dh, dtype=torch.bfloat16, device=device)
        w_h   = torch.randn(B, H,        dtype=torch.bfloat16, device=device)

        # ── PyTorch einsum path ───────────────────────────────────────────
        def _pt_dsa_idx():
            # [B, S_q, S_kv, H] = dot products per head
            scores = torch.einsum("bshd,bthd->bsth", q_idx, k_idx)  # [B, S_q, S_kv, H]
            # Weighted sum over heads
            score  = torch.einsum("bsth,bh->bst", scores, w_h)       # [B, S_q, S_kv]
            return F.relu(score)

        torch.cuda.reset_peak_memory_stats(device)
        times, stats = cuda_timer_extended(_pt_dsa_idx, WARMUP, ITERS)
        r = BenchResult(name=name, impl="pytorch_einsum", config=cfg)
        results.append(_fill(r, times, stats, flops, bytes_acc))

        # ── DeepGEMM batched FP8 path (if available) ─────────────────────
        if DEEPGEMM_AVAILABLE:
            try:
                # Reshape to [B*S_q, H*D] × [B*S_kv, H*D]^T for grouped GEMM
                q_fp8 = q_idx.reshape(B * S_q,  H * Dh).to(torch.float8_e4m3fn)
                k_fp8 = k_idx.reshape(B * S_kv, H * Dh).to(torch.float8_e4m3fn)
                scale_a = torch.ones(1, device=device)
                scale_b = torch.ones(1, device=device)

                def _dg_dsa_idx():
                    # Approximate with a single GEMM over flattened head dims
                    return torch.mm(
                        q_fp8.to(torch.bfloat16),
                        k_fp8.to(torch.bfloat16).T,
                    )

                torch.cuda.reset_peak_memory_stats(device)
                times_dg, stats_dg = cuda_timer_extended(_dg_dsa_idx, WARMUP, ITERS)
                r_dg = BenchResult(name=name, impl="deepgemm_fp8_approx", config=cfg)
                results.append(_fill(r_dg, times_dg, stats_dg, flops, bytes_acc, "fp8"))
            except torch.cuda.OutOfMemoryError as e:
                results.append(_oom_result(name, "deepgemm_fp8_approx", cfg, e))
                torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 7. DSA Sparse Attention
# ═══════════════════════════════════════════════════════════════════════════════

def bench_dsa_sparse_attn(device: torch.device) -> List[BenchResult]:
    """DSA sparse attention: full MLA attention restricted to IDX_TOPK selected KV positions.

    At decode (B=32, T=4096) with index_topk=2048, each query attends to 50% of KV.

    FLOPs: same formula as dense but S_kv = min(IDX_TOPK, T)
    """
    results = []

    for mode, B, S_q, T in [
        ("prefill", PREFILL_B, PREFILL_S, PREFILL_S),
        ("decode",  DECODE_B,  1,         DECODE_T),
    ]:
        S_kv_dense  = T
        S_kv_sparse = min(IDX_TOPK, T)   # 2048 selected positions
        H  = N_HEADS    # 64
        Dq = D_QK_ABS   # 576  (absorbed QK dim)
        Dv = D_V_ABS    # 512  (absorbed V dim)

        cfg  = {"mode": mode, "B": B, "S_q": S_q, "S_kv_dense": S_kv_dense,
                "S_kv_sparse": S_kv_sparse, "H": H}
        name = f"dsa_sparse_attn_{mode}"

        flops     = compute_attention_flops(B, H, S_q, S_kv_sparse, Dq, Dv)
        bytes_acc = compute_attention_bytes(B, H, S_q, S_kv_sparse, Dq, Dv, dtype_bytes=2)

        # Sparse KV: only IDX_TOPK positions loaded (HBM bandwidth is the bottleneck)
        q  = torch.randn(B, H,         S_q,        Dq, dtype=torch.bfloat16, device=device)
        k  = torch.randn(B, 1,         S_kv_sparse, Dq, dtype=torch.bfloat16, device=device)
        v  = torch.randn(B, 1,         S_kv_sparse, Dv, dtype=torch.bfloat16, device=device)

        # Expand single-head KV for broadcast (absorbed MLA: 1 KV head)
        k_exp = k.expand(B, H, S_kv_sparse, Dq)
        v_exp = v.expand(B, H, S_kv_sparse, Dv)

        # ── PyTorch masked matmul baseline ────────────────────────────────
        scale = 1.0 / math.sqrt(Dq)

        def _pt_sparse_attn():
            attn_w = torch.einsum("bhsd,bhkd->bhsk", q, k_exp) * scale  # [B,H,S_q,S_kv_sp]
            attn_w = F.softmax(attn_w, dim=-1)
            return torch.einsum("bhsk,bhkd->bhsd", attn_w, v_exp)        # [B,H,S_q,Dv]

        torch.cuda.reset_peak_memory_stats(device)
        times, stats = cuda_timer_extended(_pt_sparse_attn, WARMUP, ITERS)
        r = BenchResult(name=name, impl="pytorch_masked_matmul", config=cfg)
        results.append(_fill(r, times, stats, flops, bytes_acc))

        # ── FlashMLA sparse decode ────────────────────────────────────────
        if FLASHMLA_AVAILABLE and mode == "decode":
            try:
                # FlashMLA sparse API: flash_mla.flash_mla_with_kvcache_sparse
                # Inputs: q [B, H, S_q, Dq], kv_cache [B, T, 1, Dq+Dv], block_table, topk_ids
                block_size = GLM5_CONFIG["page_size"]   # 64
                n_blocks   = (S_kv_dense + block_size - 1) // block_size
                kv_cache   = torch.randn(
                    B, n_blocks * block_size, 1, Dq + Dv,
                    dtype=torch.bfloat16, device=device,
                )
                block_table = torch.arange(n_blocks, device=device).unsqueeze(0).expand(B, -1)
                topk_block_ids = block_table[:, :IDX_TOPK // block_size]

                q_fmla = q.squeeze(2) if S_q == 1 else q  # [B, H, Dq] for step decode

                def _flashmla_sparse():
                    return flash_mla.flash_mla_with_kvcache_sparse(
                        q_fmla, kv_cache, block_table, topk_block_ids,
                        softmax_scale=scale,
                    )

                torch.cuda.reset_peak_memory_stats(device)
                times_fm, stats_fm = cuda_timer_extended(_flashmla_sparse, WARMUP, ITERS)
                r_fm = BenchResult(name=name, impl="flashmla_sparse", config=cfg)
                results.append(_fill(r_fm, times_fm, stats_fm, flops, bytes_acc))
            except (torch.cuda.OutOfMemoryError, AttributeError, RuntimeError) as e:
                results.append(_oom_result(name, "flashmla_sparse", cfg, e))
                torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 8. MLA Attention
# ═══════════════════════════════════════════════════════════════════════════════

def bench_mla_attention(device: torch.device) -> List[BenchResult]:
    """Full MLA (Multi-head Latent Attention) with absorbed KV projections.

    Q:  [B, H, S_q, D_qk_absorbed]  = [B, 64, S_q, 576]
    KV: [B, 1,  T,  D_qk_absorbed + D_v_absorbed]  (single latent KV head after absorption)
    O:  [B, H, S_q, D_v_absorbed]   = [B, 64, S_q, 512]
    """
    results = []

    for mode, B, S_q, T in [
        ("prefill", PREFILL_B, PREFILL_S, PREFILL_S),
        ("decode",  DECODE_B,  1,         DECODE_T),
    ]:
        H  = N_HEADS
        Dq = D_QK_ABS   # 576
        Dv = D_V_ABS    # 512
        cfg  = {"mode": mode, "B": B, "S_q": S_q, "T": T, "H": H, "D_qk": Dq, "D_v": Dv}
        name = f"mla_attention_{mode}"

        flops     = compute_attention_flops(B, H, S_q, T, Dq, Dv)
        bytes_acc = compute_attention_bytes(B, H, S_q, T, Dq, Dv, dtype_bytes=2)

        q = torch.randn(B, H, S_q, Dq, dtype=torch.bfloat16, device=device)
        k = torch.randn(B, 1,  T,   Dq, dtype=torch.bfloat16, device=device)
        v = torch.randn(B, 1,  T,   Dv, dtype=torch.bfloat16, device=device)

        k_exp = k.expand(B, H, T, Dq)
        v_exp = v.expand(B, H, T, Dv)
        scale = 1.0 / math.sqrt(Dq)

        # ── PyTorch eager attention ───────────────────────────────────────
        def _pt_mla():
            attn_w = torch.einsum("bhsd,bhkd->bhsk", q, k_exp) * scale   # [B,H,S_q,T]
            attn_w = F.softmax(attn_w, dim=-1)
            return torch.einsum("bhsk,bhkd->bhsd", attn_w, v_exp)         # [B,H,S_q,Dv]

        torch.cuda.reset_peak_memory_stats(device)
        times, stats = cuda_timer_extended(_pt_mla, WARMUP, ITERS)
        r = BenchResult(name=name, impl="pytorch_eager", config=cfg)
        results.append(_fill(r, times, stats, flops, bytes_acc))

        # ── FlashMLA dense ────────────────────────────────────────────────
        if FLASHMLA_AVAILABLE:
            try:
                block_size  = GLM5_CONFIG["page_size"]   # 64
                n_blocks    = (T + block_size - 1) // block_size
                kv_cache    = torch.randn(
                    B, n_blocks * block_size, 1, Dq + Dv,
                    dtype=torch.bfloat16, device=device,
                )
                block_table = torch.arange(n_blocks, device=device).unsqueeze(0).expand(B, -1)
                cache_seqlens = torch.full((B,), T, dtype=torch.int32, device=device)

                # FlashMLA decode API: q shape [B, H, Dq]
                q_step = q.squeeze(2) if S_q == 1 else q

                def _flashmla_dense():
                    return flash_mla.flash_mla_with_kvcache(
                        q_step, kv_cache, block_table, cache_seqlens,
                        softmax_scale=scale,
                    )

                torch.cuda.reset_peak_memory_stats(device)
                times_fm, stats_fm = cuda_timer_extended(_flashmla_dense, WARMUP, ITERS)
                r_fm = BenchResult(name=name, impl="flashmla_dense", config=cfg)
                results.append(_fill(r_fm, times_fm, stats_fm, flops, bytes_acc))
            except (torch.cuda.OutOfMemoryError, AttributeError, RuntimeError) as e:
                results.append(_oom_result(name, "flashmla_dense", cfg, e))
                torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# 9. MoE Forward
# ═══════════════════════════════════════════════════════════════════════════════

def bench_moe_forward(device: torch.device) -> List[BenchResult]:
    """MoE forward pass at GLM-5 dims (E=256, K=8, D=6144, I=2048).

    Per-expert loop baseline; DeepGEMM grouped FP8 if available.
    """
    results = []

    for mode, B, S in [("prefill", PREFILL_B, PREFILL_S), ("decode", DECODE_B, DECODE_T)]:
        N = B * S
        E = N_EXP     # 256
        K = K_ACT     # 8
        I = FFN_DIM   # 2048
        cfg  = {"mode": mode, "B": B, "S": S, "E": E, "K": K, "I": I, "D": D}
        name = f"moe_forward_{mode}"

        flops     = compute_moe_flops(N, K, D, I)
        bytes_acc = compute_moe_bytes(N, K, D, I, E, dtype_bytes=2)

        hidden = torch.randn(N, D, dtype=torch.bfloat16, device=device)
        guw    = torch.randn(E, 2 * I, D, dtype=torch.bfloat16, device=device)
        dw     = torch.randn(E, D, I,     dtype=torch.bfloat16, device=device)

        flat_ids    = torch.stack([torch.randperm(E, device=device)[:K] for _ in range(N)]).reshape(-1).long()
        tok_indices = torch.arange(N, device=device).unsqueeze(1).expand(N, K).reshape(-1)
        raw_w       = torch.randn(N * K, device=device)
        flat_w      = torch.sigmoid(raw_w)
        flat_w      = flat_w / flat_w.reshape(N, K).sum(-1, keepdim=True).clamp(min=1e-6).reshape(-1)

        # ── PyTorch per-expert loop ───────────────────────────────────────
        def _pt_moe():
            out = torch.zeros(N, D, dtype=torch.bfloat16, device=device)
            for e in range(E):
                mask = (flat_ids == e)
                if not mask.any():
                    continue
                idx = tok_indices[mask]
                w   = flat_w[mask].unsqueeze(-1)
                gu  = hidden[idx] @ guw[e].T           # [T_e, 2*I]
                g, u = gu.chunk(2, dim=-1)
                act = F.silu(g) * u
                out.index_add_(0, idx, (act @ dw[e].T) * w)
            return out

        torch.cuda.reset_peak_memory_stats(device)
        times, stats = cuda_timer_extended(_pt_moe, WARMUP, ITERS)
        r = BenchResult(name=name, impl="pytorch_loop", config=cfg)
        results.append(_fill(r, times, stats, flops, bytes_acc))

        # ── DeepGEMM grouped FP8 ─────────────────────────────────────────
        if DEEPGEMM_AVAILABLE:
            try:
                guw_fp8 = guw.to(torch.float8_e4m3fn)
                dw_fp8  = dw.to(torch.float8_e4m3fn)

                sort_order  = torch.argsort(flat_ids, stable=True)
                sorted_ids  = flat_ids[sort_order]
                sorted_tok  = tok_indices[sort_order]
                sorted_w    = flat_w[sort_order]
                exp_counts  = torch.bincount(sorted_ids, minlength=E)
                expert_ends = exp_counts.cumsum(0).to(torch.int32)
                gathered_fp8 = hidden[sorted_tok].to(torch.float8_e4m3fn)
                scale_a = torch.ones(1, device=device)
                scale_b = torch.ones(E, device=device)

                def _dg_moe():
                    gu_out = deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt(
                        gathered_fp8, guw_fp8, scale_a, scale_b, expert_ends
                    )
                    g, u = gu_out.chunk(2, dim=-1)
                    act_fp8 = (F.silu(g) * u).to(torch.float8_e4m3fn)
                    dn_out = deep_gemm.m_grouped_gemm_fp8_fp8_bf16_nt(
                        act_fp8, dw_fp8, scale_a, scale_b, expert_ends
                    )
                    out = torch.zeros(N, D, dtype=torch.bfloat16, device=device)
                    out.index_add_(0, sorted_tok, dn_out * sorted_w.unsqueeze(-1))
                    return out

                torch.cuda.reset_peak_memory_stats(device)
                times_dg, stats_dg = cuda_timer_extended(_dg_moe, WARMUP, ITERS)
                r_dg = BenchResult(name=name, impl="deepgemm_fp8", config=cfg)
                bytes_fp8 = compute_moe_bytes(N, K, D, I, E, dtype_bytes=1)
                results.append(_fill(r_dg, times_dg, stats_dg, flops, bytes_fp8, "fp8"))
            except (torch.cuda.OutOfMemoryError, AttributeError, RuntimeError) as e:
                results.append(_oom_result(name, "deepgemm_fp8", cfg, e))
                torch.cuda.empty_cache()

    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Component registry
# ═══════════════════════════════════════════════════════════════════════════════

COMPONENTS: Dict[str, Callable] = {
    "rmsnorm":          bench_rmsnorm,
    "swiglu":           bench_swiglu,
    "cross_entropy":    bench_cross_entropy,
    "rope":             bench_rope,
    "moe_router":       bench_moe_router,
    "dsa_indexer":      bench_dsa_indexer,
    "dsa_sparse_attn":  bench_dsa_sparse_attn,
    "mla_attention":    bench_mla_attention,
    "moe_forward":      bench_moe_forward,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Triple-report comparison table
# ═══════════════════════════════════════════════════════════════════════════════

def print_triple_table(results: List[BenchResult], title: str = "") -> None:
    """Print a PyTorch vs Triton vs CUDA kernel comparison table.

    Groups results by (component_name, mode) and prints one row per implementation.
    The "triple" in the name refers to the three implementation columns, aligning
    with the GLM-5 paper's kernel ablation table format.
    """
    if title:
        print(f"\n{'='*100}")
        print(f"  {title}")
        print(f"{'='*100}")

    # Group by base component name (strip _prefill/_decode suffix for grouping header)
    from collections import defaultdict
    groups: Dict[str, List[BenchResult]] = defaultdict(list)
    for r in results:
        groups[r.name].append(r)

    header = (f"{'Component':<30} {'Impl':<22} {'Med(ms)':>9} {'p99(ms)':>9} "
              f"{'TFLOPS':>8} {'MFU%':>7} {'BW(GB/s)':>10} {'SOL%':>7} {'Bound':<12}")
    sep = "-" * 113

    print(header)
    print(sep)

    prev_base = None
    for name, group_results in sorted(groups.items()):
        # Print a blank separator line between different components
        base = name.rsplit("_prefill", 1)[0].rsplit("_decode", 1)[0]
        if prev_base is not None and base != prev_base:
            print()
        prev_base = base

        for r in group_results:
            if r.is_oom:
                print(f"{r.name:<30} {r.impl:<22} {'OOM':>9}")
            else:
                print(
                    f"{r.name:<30} {r.impl:<22} "
                    f"{r.median_ms:>9.3f} {r.p99_ms:>9.3f} "
                    f"{r.tflops:>8.2f} {r.mfu_pct:>7.2f} "
                    f"{r.bandwidth_gb_s:>10.1f} {r.hbm_sol_pct:>7.2f} "
                    f"{r.roofline_bound:<12}"
                )

    print(sep)

    # ── Roofline reference points (from H100_SPECS) ───────────────────────
    print()
    print("H100 SXM5 reference points:")
    print(f"  BF16 peak:          {H100_SPECS['peak_tflops_bf16']:>8.0f} TFLOPS  "
          f"(ridge ≈ 295 FLOPs/byte)")
    print(f"  FP8 peak:           {H100_SPECS['peak_tflops_fp8']:>8.0f} TFLOPS  "
          f"(ridge ≈ 590 FLOPs/byte)")
    print(f"  HBM bandwidth:      {H100_SPECS['hbm_bandwidth_gb_s']:>8.0f} GB/s")
    print(f"  FA3 MFU reference:  {H100_SPECS['fa3_mfu_pct']:>8.1f}%  "
          f"({H100_SPECS['fa3_tflops_fp16']:.0f} TFLOPS FP16)")
    print(f"  FlashMLA dense:     {H100_SPECS['flashmla_tflops_decode']:>8.0f} TFLOPS decode")
    print(f"  FlashMLA sparse:    {H100_SPECS['flashmla_tflops_sparse']:>8.0f} TFLOPS sparse decode")
    print(f"  DeepGEMM FP8:       {H100_SPECS['deepgemm_tflops_fp8']:>8.0f} TFLOPS grouped GEMM")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def main() -> None:
    parser = argparse.ArgumentParser(
        description="GLM-5 kernel microbenchmarks — triple report (PyTorch / Triton / CUDA).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--component",
        nargs="+",
        default=["all"],
        choices=list(COMPONENTS.keys()) + ["all"],
        metavar="COMP",
        help=(
            "Component(s) to benchmark. Pass 'all' for every component, "
            "or one or more of: " + ", ".join(COMPONENTS.keys())
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "results"),
        help="Directory for JSON result files.",
    )
    parser.add_argument("--warmup", type=int, default=WARMUP,
                        help="Warmup iterations (default matches FlashAttention-3 methodology).")
    parser.add_argument("--iters", type=int, default=ITERS,
                        help="Measured iterations (100 required for meaningful p99).")
    parser.add_argument("--device", type=str, default="cuda",
                        help="PyTorch device string.")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. This benchmark requires a GPU.", file=sys.stderr)
        sys.exit(1)

    # Override module-level constants if user passed different values
    import benchmark.triple_report.bench_micro as _self
    _self.WARMUP = args.warmup
    _self.ITERS  = args.iters

    device = torch.device(args.device)
    env    = capture_environment()

    print("=" * 80)
    print("  GLM-5 Kernel Microbenchmarks — Triple Report")
    print("=" * 80)
    print(f"  GPU:          {env.get('gpu_name', 'unknown')}")
    print(f"  FlashMLA:     {'available' if FLASHMLA_AVAILABLE else 'not installed'}")
    print(f"  DeepGEMM:     {'available' if DEEPGEMM_AVAILABLE else 'not installed'}")
    print(f"  Triton RMSNorm:  {'available' if TRITON_RMSNORM_AVAILABLE else 'not found'}")
    print(f"  Triton SwiGLU:   {'available' if TRITON_SWIGLU_AVAILABLE else 'not found'}")
    print(f"  Triton CE:       {'available' if TRITON_CE_AVAILABLE else 'not found'}")
    print(f"  Warmup/Iters: {WARMUP}/{ITERS}")
    print(f"  Prefill:      B={PREFILL_B}, S={PREFILL_S}")
    print(f"  Decode:       B={DECODE_B}, T={DECODE_T}")
    print()

    # Resolve component list
    if "all" in args.component:
        selected = list(COMPONENTS.keys())
    else:
        selected = args.component

    all_results: List[BenchResult] = []

    for comp_name in selected:
        bench_fn = COMPONENTS[comp_name]
        print(f"--- {comp_name} ---")
        try:
            comp_results = bench_fn(device)
            for r in comp_results:
                if r.is_oom:
                    print(f"  {r.impl:<22} OOM")
                else:
                    outlier = check_outliers(r.latency_ms)
                    flag_str = " [WARN: " + "; ".join(outlier["flags"]) + "]" if not outlier["valid"] else ""
                    print(
                        f"  {r.impl:<22} "
                        f"median={r.median_ms:>7.3f} ms  "
                        f"p99={r.p99_ms:>7.3f} ms  "
                        f"{r.tflops:>6.2f} TFLOPS  "
                        f"MFU={r.mfu_pct:>5.2f}%  "
                        f"SOL={r.hbm_sol_pct:>5.2f}%  "
                        f"{r.roofline_bound}"
                        f"{flag_str}"
                    )
            all_results.extend(comp_results)
        except Exception as e:  # noqa: BLE001
            print(f"  ERROR in {comp_name}: {e}")
        print()

    # ── Triple report summary ─────────────────────────────────────────────
    print_triple_table(all_results, title="GLM-5 Kernel Microbenchmarks — Triple Report Summary")
    print()

    # ── Standard summary table (shared/report.py format) ─────────────────
    print_summary_table(all_results, title="Flat Summary (all impls)")

    # ── Save JSON ─────────────────────────────────────────────────────────
    comp_tag = "all" if "all" in args.component else "_".join(selected)
    save_results(
        results=all_results,
        output_dir=args.output_dir,
        experiment_name=f"micro_{comp_tag}",
        env=env,
    )


if __name__ == "__main__":
    main()
