#!/usr/bin/env python3
"""MLA Roofline Decomposition Benchmark.

Decomposes MLA into its 6 sub-operations and measures each independently
on the H100 roofline. Compares MLA (compressed KV) against standard MHA
(full-dimension KV) to show where MLA's latent compression changes the
memory-compute tradeoff.

Usage:
    python3 -m benchmark.mla_roofline.bench_mla_roofline
    python3 -m benchmark.mla_roofline.bench_mla_roofline --quick
    python3 -m benchmark.mla_roofline.bench_mla_roofline --output-dir results/roofline/

Output:
    - JSON with per-sub-op timing, FLOPs, bytes, OI, TFLOPS, bound classification
    - Human-readable roofline summary table
    - Data suitable for plotting the classic roofline diagram
"""

import argparse
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from benchmark.shared.config import GLM5_CONFIG, H100_SPECS
from benchmark.shared.timer import cuda_timer_extended, check_outliers
from benchmark.shared.metrics import (
    compute_tflops,
    compute_bandwidth_gb_s,
    compute_operational_intensity,
    compute_mfu,
    compute_hbm_sol,
    classify_roofline_bound,
    compute_roofline_achievable,
)
from benchmark.shared.report import save_results, capture_environment


# ─────────────────────────────────────────────────────────────────────────
# Sub-operation FLOP and byte count models
# ─────────────────────────────────────────────────────────────────────────

def _linear_flops(M: int, N: int, K: int) -> int:
    """FLOPs for a linear layer (matmul): 2*M*N*K."""
    return 2 * M * N * K


def _linear_bytes(M: int, N: int, K: int, dtype_bytes: int = 2) -> int:
    """Bytes for a linear layer: read input [M,K] + weight [K,N] + write output [M,N]."""
    return (M * K + K * N + M * N) * dtype_bytes


def _attention_flops(B: int, H: int, S_q: int, S_kv: int,
                     d_qk: int, d_v: int) -> int:
    """FLOPs for core attention: QK^T + softmax + attn×V.

    QK^T:    2 * B * H * S_q * S_kv * d_qk
    softmax: ~5 * B * H * S_q * S_kv  (negligible)
    attn×V:  2 * B * H * S_q * S_kv * d_v
    """
    return 2 * B * H * S_q * S_kv * (d_qk + d_v)


def _attention_bytes(B: int, H: int, S_q: int, S_kv: int,
                     d_qk: int, d_v: int, n_kv_heads: int,
                     dtype_bytes: int = 2) -> int:
    """Bytes for core attention.

    Read:  Q [B, H, S_q, d_qk]
           K [B, n_kv_heads, S_kv, d_qk]  (MLA: 1 KV head; MHA: H KV heads)
           V [B, n_kv_heads, S_kv, d_v]
    Write: O [B, H, S_q, d_v]
    """
    q_bytes = B * H * S_q * d_qk * dtype_bytes
    k_bytes = B * n_kv_heads * S_kv * d_qk * dtype_bytes
    v_bytes = B * n_kv_heads * S_kv * d_v * dtype_bytes
    o_bytes = B * H * S_q * d_v * dtype_bytes
    return q_bytes + k_bytes + v_bytes + o_bytes


# ─────────────────────────────────────────────────────────────────────────
# Sub-operation definitions for MLA
# ─────────────────────────────────────────────────────────────────────────

@dataclass
class SubOp:
    """A single sub-operation to benchmark."""
    name: str
    description: str
    flops: int = 0
    bytes_accessed: int = 0
    # Measured
    median_ms: float = 0.0
    tflops: float = 0.0
    bandwidth_gb_s: float = 0.0
    operational_intensity: float = 0.0
    roofline_bound: str = ""
    mfu_pct: float = 0.0
    hbm_sol_pct: float = 0.0
    roofline_achievable_tflops: float = 0.0
    efficiency_pct: float = 0.0  # achieved / roofline_achievable
    ci_95: List[float] = field(default_factory=list)
    cv: float = 0.0


def build_mla_subops(B: int, S_q: int, S_kv: int, cfg: dict) -> List[SubOp]:
    """Define MLA sub-operations with analytical FLOP/byte counts.

    GLM-5 MLA dimensions:
      hidden_size:      6144
      q_lora_rank:      2048
      kv_lora_rank:     512
      qk_rope_head_dim: 64
      qk_nope_head_dim: 192
      qk_head_dim:      256  (192 + 64)
      v_head_dim:       256
      num_heads:        64
    """
    H = cfg["num_heads"]
    D = cfg["hidden_size"]
    q_lora = cfg["q_lora_rank"]
    kv_lora = cfg["kv_lora_rank"]
    qk_rope = cfg["qk_rope_head_dim"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_head = cfg["qk_head_dim"]  # nope + rope
    v_head = cfg["v_head_dim"]

    N = B * S_q  # total query tokens

    ops = []

    # 1. q_a_proj: [N, D] × [D, q_lora] → [N, q_lora]
    ops.append(SubOp(
        name="q_a_proj",
        description=f"Query compression: [{N},{D}] × [{D},{q_lora}]",
        flops=_linear_flops(N, q_lora, D),
        bytes_accessed=_linear_bytes(N, q_lora, D),
    ))

    # 2. q_b_proj: [N, q_lora] × [q_lora, H*qk_head] → [N, H*qk_head]
    q_out = H * qk_head
    ops.append(SubOp(
        name="q_b_proj",
        description=f"Query expansion: [{N},{q_lora}] × [{q_lora},{q_out}]",
        flops=_linear_flops(N, q_out, q_lora),
        bytes_accessed=_linear_bytes(N, q_out, q_lora),
    ))

    # 3. kv_a_proj: [N, D] × [D, kv_lora+rope] → [N, kv_lora+rope]
    kv_out = kv_lora + qk_rope
    N_kv = B * S_kv  # KV tokens (different from Q tokens in decode)
    ops.append(SubOp(
        name="kv_a_proj",
        description=f"KV compression: [{N_kv},{D}] × [{D},{kv_out}]",
        flops=_linear_flops(N_kv, kv_out, D),
        bytes_accessed=_linear_bytes(N_kv, kv_out, D),
    ))

    # 4. kv_b_proj: [N_kv, kv_lora] × [kv_lora, H*(nope+v)] → [N_kv, H*(nope+v)]
    kv_expand_out = H * (qk_nope + v_head)
    ops.append(SubOp(
        name="kv_b_proj",
        description=f"KV expansion: [{N_kv},{kv_lora}] × [{kv_lora},{kv_expand_out}]",
        flops=_linear_flops(N_kv, kv_expand_out, kv_lora),
        bytes_accessed=_linear_bytes(N_kv, kv_expand_out, kv_lora),
    ))

    # 5. core attention: Q×K^T + softmax + attn×V
    ops.append(SubOp(
        name="attention",
        description=f"QK^T + softmax + attn×V: [{B},{H},{S_q},{qk_head}] × [{B},{H},{S_kv},{qk_head}]",
        flops=_attention_flops(B, H, S_q, S_kv, qk_head, v_head),
        bytes_accessed=_attention_bytes(B, H, S_q, S_kv, qk_head, v_head,
                                        n_kv_heads=H, dtype_bytes=2),
    ))

    # 6. o_proj: [N, H*v_head] × [H*v_head, D] → [N, D]
    o_in = H * v_head
    ops.append(SubOp(
        name="o_proj",
        description=f"Output projection: [{N},{o_in}] × [{o_in},{D}]",
        flops=_linear_flops(N, D, o_in),
        bytes_accessed=_linear_bytes(N, D, o_in),
    ))

    return ops


def build_mha_subops(B: int, S_q: int, S_kv: int, cfg: dict) -> List[SubOp]:
    """Define standard MHA sub-operations for comparison.

    Standard MHA (no compression):
      Q proj: [N, D] → [N, H*d_head]
      K proj: [N, D] → [N, H*d_head]  (full per-head KV, no compression)
      V proj: [N, D] → [N, H*d_head]
      attention: same as MLA but with full-dim KV
      O proj: same as MLA
    """
    H = cfg["num_heads"]
    D = cfg["hidden_size"]
    d_head = D // H  # 6144 // 64 = 96 for standard MHA

    N = B * S_q
    N_kv = B * S_kv

    ops = []

    # Q projection (single matmul, no LoRA)
    ops.append(SubOp(
        name="q_proj",
        description=f"Q projection: [{N},{D}] × [{D},{H * d_head}]",
        flops=_linear_flops(N, H * d_head, D),
        bytes_accessed=_linear_bytes(N, H * d_head, D),
    ))

    # K projection
    ops.append(SubOp(
        name="k_proj",
        description=f"K projection: [{N_kv},{D}] × [{D},{H * d_head}]",
        flops=_linear_flops(N_kv, H * d_head, D),
        bytes_accessed=_linear_bytes(N_kv, H * d_head, D),
    ))

    # V projection
    ops.append(SubOp(
        name="v_proj",
        description=f"V projection: [{N_kv},{D}] × [{D},{H * d_head}]",
        flops=_linear_flops(N_kv, H * d_head, D),
        bytes_accessed=_linear_bytes(N_kv, H * d_head, D),
    ))

    # Attention (full per-head KV — H KV heads, not 1)
    ops.append(SubOp(
        name="attention",
        description=f"QK^T + softmax + attn×V: [{B},{H},{S_q},{d_head}] × [{B},{H},{S_kv},{d_head}]",
        flops=_attention_flops(B, H, S_q, S_kv, d_head, d_head),
        bytes_accessed=_attention_bytes(B, H, S_q, S_kv, d_head, d_head,
                                        n_kv_heads=H, dtype_bytes=2),
    ))

    # O projection
    ops.append(SubOp(
        name="o_proj",
        description=f"Output projection: [{N},{H * d_head}] × [{H * d_head},{D}]",
        flops=_linear_flops(N, D, H * d_head),
        bytes_accessed=_linear_bytes(N, D, H * d_head),
    ))

    return ops


# ─────────────────────────────────────────────────────────────────────────
# Benchmark runners
# ─────────────────────────────────────────────────────────────────────────

def _make_linear(in_f: int, out_f: int, device: str, dtype: torch.dtype) -> nn.Linear:
    """Create a linear layer with random weights."""
    layer = nn.Linear(in_f, out_f, bias=False, device=device, dtype=dtype)
    return layer


def benchmark_mla_subops(
    B: int, S_q: int, S_kv: int, cfg: dict,
    device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
    warmup: int = 10, iters: int = 100,
) -> List[SubOp]:
    """Benchmark each MLA sub-operation independently."""
    H = cfg["num_heads"]
    D = cfg["hidden_size"]
    q_lora = cfg["q_lora_rank"]
    kv_lora = cfg["kv_lora_rank"]
    qk_rope = cfg["qk_rope_head_dim"]
    qk_nope = cfg["qk_nope_head_dim"]
    qk_head = cfg["qk_head_dim"]
    v_head = cfg["v_head_dim"]

    N = B * S_q
    N_kv = B * S_kv

    subops = build_mla_subops(B, S_q, S_kv, cfg)

    # Create tensors and layers
    hidden_q = torch.randn(N, D, device=device, dtype=dtype)
    hidden_kv = torch.randn(N_kv, D, device=device, dtype=dtype)

    q_a = _make_linear(D, q_lora, device, dtype)
    q_b = _make_linear(q_lora, H * qk_head, device, dtype)
    kv_a = _make_linear(D, kv_lora + qk_rope, device, dtype)
    kv_b = _make_linear(kv_lora, H * (qk_nope + v_head), device, dtype)
    o_proj = _make_linear(H * v_head, D, device, dtype)

    # Pre-compute intermediates for dependent ops
    with torch.no_grad():
        q_compressed = q_a(hidden_q)                          # [N, q_lora]
        q_expanded = q_b(q_compressed)                        # [N, H*qk_head]
        kv_compressed_full = kv_a(hidden_kv)                  # [N_kv, kv_lora+rope]
        kv_compressed = kv_compressed_full[:, :kv_lora]       # [N_kv, kv_lora]
        kv_expanded = kv_b(kv_compressed)                     # [N_kv, H*(nope+v)]

        # Reshape for attention
        Q = q_expanded.view(B, S_q, H, qk_head).transpose(1, 2)    # [B,H,S_q,qk_head]
        KV = kv_expanded.view(B, S_kv, H, qk_nope + v_head)
        K = KV[..., :qk_nope + qk_rope].transpose(1, 2)            # approximate K dims
        # For timing, we just need correctly-shaped tensors
        K = torch.randn(B, H, S_kv, qk_head, device=device, dtype=dtype)
        V = torch.randn(B, H, S_kv, v_head, device=device, dtype=dtype)
        attn_out = torch.randn(N, H * v_head, device=device, dtype=dtype)

    # ── Benchmark each sub-op ──
    fns = [
        lambda: q_a(hidden_q),
        lambda: q_b(q_compressed),
        lambda: kv_a(hidden_kv),
        lambda: kv_b(kv_compressed),
        None,  # attention — handled separately
        lambda: o_proj(attn_out),
    ]

    for i, (op, fn) in enumerate(zip(subops, fns)):
        if op.name == "attention":
            # Attention needs special handling — can OOM at large S_kv
            scaling = qk_head ** -0.5
            try:
                def attn_fn():
                    scores = torch.matmul(Q, K.transpose(2, 3)) * scaling
                    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
                    return torch.matmul(weights, V)

                times, stats = cuda_timer_extended(attn_fn, warmup=warmup, iters=iters)
            except torch.cuda.OutOfMemoryError:
                # Attention OOMs — record it but continue
                op.median_ms = -1.0
                op.roofline_bound = "OOM"
                torch.cuda.empty_cache()
                continue
        else:
            times, stats = cuda_timer_extended(fn, warmup=warmup, iters=iters)

        op.median_ms = stats["median"]
        op.ci_95 = [stats["ci_95_low"], stats["ci_95_high"]]
        op.cv = stats["cv"]

        latency_s = stats["median"] / 1000.0
        op.tflops = compute_tflops(op.flops, latency_s)
        op.bandwidth_gb_s = compute_bandwidth_gb_s(op.bytes_accessed, latency_s)
        op.operational_intensity = compute_operational_intensity(op.flops, op.bytes_accessed)
        op.roofline_bound = classify_roofline_bound(op.operational_intensity, "bf16")
        op.mfu_pct = compute_mfu(op.flops, latency_s, "bf16")
        op.hbm_sol_pct = compute_hbm_sol(op.bytes_accessed, latency_s)
        op.roofline_achievable_tflops = compute_roofline_achievable(op.operational_intensity, "bf16")
        if op.roofline_achievable_tflops > 0:
            op.efficiency_pct = (op.tflops / op.roofline_achievable_tflops) * 100

        # Check for timing anomalies
        outlier_info = check_outliers(times)
        if not outlier_info["valid"]:
            op.description += f" [WARNING: {outlier_info['flags']}]"

    return subops


def benchmark_mha_subops(
    B: int, S_q: int, S_kv: int, cfg: dict,
    device: str = "cuda", dtype: torch.dtype = torch.bfloat16,
    warmup: int = 10, iters: int = 100,
) -> List[SubOp]:
    """Benchmark standard MHA sub-operations for comparison."""
    H = cfg["num_heads"]
    D = cfg["hidden_size"]
    d_head = D // H

    N = B * S_q
    N_kv = B * S_kv

    subops = build_mha_subops(B, S_q, S_kv, cfg)

    hidden_q = torch.randn(N, D, device=device, dtype=dtype)
    hidden_kv = torch.randn(N_kv, D, device=device, dtype=dtype)

    q_proj = _make_linear(D, H * d_head, device, dtype)
    k_proj = _make_linear(D, H * d_head, device, dtype)
    v_proj = _make_linear(D, H * d_head, device, dtype)
    o_proj = _make_linear(H * d_head, D, device, dtype)

    with torch.no_grad():
        Q = torch.randn(B, H, S_q, d_head, device=device, dtype=dtype)
        K = torch.randn(B, H, S_kv, d_head, device=device, dtype=dtype)
        V = torch.randn(B, H, S_kv, d_head, device=device, dtype=dtype)
        attn_out = torch.randn(N, H * d_head, device=device, dtype=dtype)

    fns = [
        lambda: q_proj(hidden_q),
        lambda: k_proj(hidden_kv),
        lambda: v_proj(hidden_kv),
        None,  # attention
        lambda: o_proj(attn_out),
    ]

    for i, (op, fn) in enumerate(zip(subops, fns)):
        if op.name == "attention":
            scaling = d_head ** -0.5
            try:
                def attn_fn():
                    scores = torch.matmul(Q, K.transpose(2, 3)) * scaling
                    weights = F.softmax(scores, dim=-1, dtype=torch.float32).to(dtype)
                    return torch.matmul(weights, V)

                times, stats = cuda_timer_extended(attn_fn, warmup=warmup, iters=iters)
            except torch.cuda.OutOfMemoryError:
                op.median_ms = -1.0
                op.roofline_bound = "OOM"
                torch.cuda.empty_cache()
                continue
        else:
            times, stats = cuda_timer_extended(fn, warmup=warmup, iters=iters)

        op.median_ms = stats["median"]
        op.ci_95 = [stats["ci_95_low"], stats["ci_95_high"]]
        op.cv = stats["cv"]

        latency_s = stats["median"] / 1000.0
        op.tflops = compute_tflops(op.flops, latency_s)
        op.bandwidth_gb_s = compute_bandwidth_gb_s(op.bytes_accessed, latency_s)
        op.operational_intensity = compute_operational_intensity(op.flops, op.bytes_accessed)
        op.roofline_bound = classify_roofline_bound(op.operational_intensity, "bf16")
        op.mfu_pct = compute_mfu(op.flops, latency_s, "bf16")
        op.hbm_sol_pct = compute_hbm_sol(op.bytes_accessed, latency_s)
        op.roofline_achievable_tflops = compute_roofline_achievable(op.operational_intensity, "bf16")
        if op.roofline_achievable_tflops > 0:
            op.efficiency_pct = (op.tflops / op.roofline_achievable_tflops) * 100

    return subops


# ─────────────────────────────────────────────────────────────────────────
# Main sweep
# ─────────────────────────────────────────────────────────────────────────

def print_roofline_table(label: str, subops: List[SubOp], S_kv: int):
    """Print a formatted roofline summary table."""
    ridge = H100_SPECS["peak_tflops_bf16"] * 1e12 / (H100_SPECS["hbm_bandwidth_gb_s"] * 1e9)

    print(f"\n  {label} (S_kv={S_kv}):")
    print(f"  {'Op':<12} {'ms':>8} {'TFLOPS':>8} {'GB/s':>8} {'OI':>8} {'Bound':>10} {'MFU%':>6} {'SOL%':>6} {'Eff%':>6}")
    print(f"  {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*8} {'-'*10} {'-'*6} {'-'*6} {'-'*6}")

    total_ms = 0.0
    for op in subops:
        if op.median_ms < 0:
            print(f"  {op.name:<12} {'OOM':>8}")
            continue
        total_ms += op.median_ms
        print(f"  {op.name:<12} {op.median_ms:>8.3f} {op.tflops:>8.1f} {op.bandwidth_gb_s:>8.0f} "
              f"{op.operational_intensity:>8.1f} {op.roofline_bound:>10} "
              f"{op.mfu_pct:>5.1f}% {op.hbm_sol_pct:>5.1f}% {op.efficiency_pct:>5.1f}%")

    print(f"  {'TOTAL':<12} {total_ms:>8.3f}")
    print(f"  Ridge point: {ridge:.0f} FLOPs/byte (OI < ridge = memory-bound)")


def run_sweep(args) -> Dict[str, Any]:
    """Run the full roofline sweep across context lengths."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("ERROR: CUDA required for roofline benchmark")
        sys.exit(1)

    cfg = GLM5_CONFIG
    dtype = torch.bfloat16

    if args.quick:
        context_lengths = [512, 2048, 8192]
        batch_sizes = [1, 32]
        warmup, iters = 5, 20
    else:
        context_lengths = [512, 1024, 2048, 4096, 8192, 16384]
        batch_sizes = [1, 32]
        warmup, iters = 10, 100

    # Prefill mode: S_q = S_kv (full sequence)
    # Decode mode: S_q = 1, S_kv = context length
    modes = [
        ("decode", 1),       # S_q=1: decode — the serving-critical path
        ("prefill", None),   # S_q=S_kv: prefill — the training-critical path
    ]

    env = capture_environment()
    all_results = []

    print("=" * 80)
    print("  MLA Roofline Decomposition Benchmark")
    print(f"  GPU: {env.get('gpu_name', 'unknown')}")
    print(f"  H100 ridge point: {H100_SPECS['peak_tflops_bf16'] * 1e12 / (H100_SPECS['hbm_bandwidth_gb_s'] * 1e9):.0f} FLOPs/byte (BF16)")
    print(f"  Context lengths: {context_lengths}")
    print(f"  Batch sizes: {batch_sizes}")
    print(f"  Iters: {iters} (warmup: {warmup})")
    print("=" * 80)

    for mode_name, fixed_s_q in modes:
        print(f"\n{'─' * 80}")
        print(f"  Mode: {mode_name.upper()}")
        print(f"{'─' * 80}")

        for B in batch_sizes:
            for S_kv in context_lengths:
                S_q = fixed_s_q if fixed_s_q is not None else S_kv

                # Skip prefill at large S_kv (OOM for eager attention)
                if mode_name == "prefill" and S_kv > 4096 and B > 1:
                    print(f"\n  SKIP B={B}, S_kv={S_kv} (prefill OOM at this size)")
                    continue

                # Memory estimate for attention scores: B * H * S_q * S_kv * 4 bytes
                attn_mem_gb = B * cfg["num_heads"] * S_q * S_kv * 4 / 1e9
                if attn_mem_gb > 60:  # leave headroom on 80GB GPU
                    print(f"\n  SKIP B={B}, S_kv={S_kv} (attention matrix ~{attn_mem_gb:.1f} GB, OOM risk)")
                    continue

                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()

                # ── MLA ──
                try:
                    mla_ops = benchmark_mla_subops(
                        B, S_q, S_kv, cfg, device, dtype, warmup, iters
                    )
                    print_roofline_table(f"MLA B={B}", mla_ops, S_kv)
                except Exception as e:
                    print(f"\n  MLA B={B} S_kv={S_kv}: ERROR {e}")
                    mla_ops = []

                torch.cuda.empty_cache()

                # ── Standard MHA ──
                try:
                    mha_ops = benchmark_mha_subops(
                        B, S_q, S_kv, cfg, device, dtype, warmup, iters
                    )
                    print_roofline_table(f"MHA B={B}", mha_ops, S_kv)
                except Exception as e:
                    print(f"\n  MHA B={B} S_kv={S_kv}: ERROR {e}")
                    mha_ops = []

                torch.cuda.empty_cache()

                # ── Summary comparison ──
                mla_total = sum(op.median_ms for op in mla_ops if op.median_ms > 0)
                mha_total = sum(op.median_ms for op in mha_ops if op.median_ms > 0)
                if mha_total > 0 and mla_total > 0:
                    print(f"\n  MLA vs MHA: {mla_total:.3f} ms vs {mha_total:.3f} ms "
                          f"({mla_total/mha_total:.2f}x)")

                # Store results
                peak_mem = torch.cuda.max_memory_allocated() / 1e9
                result = {
                    "mode": mode_name,
                    "B": B, "S_q": S_q, "S_kv": S_kv,
                    "mla": [asdict(op) for op in mla_ops],
                    "mha": [asdict(op) for op in mha_ops],
                    "mla_total_ms": mla_total,
                    "mha_total_ms": mha_total,
                    "mla_vs_mha": mla_total / mha_total if mha_total > 0 else None,
                    "peak_memory_gb": peak_mem,
                }
                all_results.append(result)

    return {
        "experiment": "mla_roofline_decomposition",
        "environment": env,
        "config": {k: v for k, v in cfg.items() if not k.startswith("_")},
        "h100_specs": {
            "peak_tflops_bf16": H100_SPECS["peak_tflops_bf16"],
            "hbm_bandwidth_gb_s": H100_SPECS["hbm_bandwidth_gb_s"],
            "ridge_point": H100_SPECS["peak_tflops_bf16"] * 1e12 / (H100_SPECS["hbm_bandwidth_gb_s"] * 1e9),
        },
        "sweep": {
            "context_lengths": context_lengths,
            "batch_sizes": batch_sizes,
            "warmup": warmup,
            "iters": iters,
        },
        "results": all_results,
    }


def main():
    parser = argparse.ArgumentParser(
        description="MLA Roofline Decomposition Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python3 -m benchmark.mla_roofline.bench_mla_roofline --quick
  python3 -m benchmark.mla_roofline.bench_mla_roofline --output-dir results/roofline/
        """,
    )
    parser.add_argument("--quick", action="store_true",
                        help="Quick run: fewer context lengths, fewer iterations")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Directory to save JSON results")
    args = parser.parse_args()

    results = run_sweep(args)

    # Save results
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(args.output_dir, f"mla_roofline_{ts}.json")
        with open(path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {path}")

    # Final summary
    print("\n" + "=" * 80)
    print("  ROOFLINE SUMMARY")
    print("=" * 80)

    ridge = H100_SPECS["peak_tflops_bf16"] * 1e12 / (H100_SPECS["hbm_bandwidth_gb_s"] * 1e9)
    print(f"\n  H100 BF16 ridge point: {ridge:.0f} FLOPs/byte")
    print(f"  Ops with OI < {ridge:.0f} are memory-bound (benefit from MLA compression)")
    print(f"  Ops with OI > {ridge:.0f} are compute-bound (MLA adds compute overhead)")

    print("\n  Key finding: Look for the crossover where kv_b_proj (MLA expansion)")
    print("  transitions from memory-bound to compute-bound as S_kv grows.")
    print("  This transition determines when weight absorption becomes critical.")

    # Identify bottleneck at each context length
    for r in results["results"]:
        if r["mode"] != "decode" or r["B"] != 32:
            continue
        if not r["mla"]:
            continue
        bottleneck = max(r["mla"], key=lambda x: x["median_ms"] if x["median_ms"] > 0 else 0)
        if bottleneck["median_ms"] > 0:
            pct = bottleneck["median_ms"] / r["mla_total_ms"] * 100 if r["mla_total_ms"] > 0 else 0
            print(f"\n  S_kv={r['S_kv']:>5}: bottleneck = {bottleneck['name']:<12} "
                  f"({pct:.0f}% of MLA time, {bottleneck['roofline_bound']})")


if __name__ == "__main__":
    main()
