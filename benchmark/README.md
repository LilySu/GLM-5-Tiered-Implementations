# GLM-5 Benchmark Suite

Academic-grade benchmark harness for H100 GPUs, aligned with MLSys/OSDI/NeurIPS 2024-2025 evaluation standards.

## Structure

```
benchmark/
  shared/                    # Shared utilities and metrics
    __init__.py
    timer.py                 # Extended CUDA timer (100 iters, bootstrap CI, p99)
    metrics.py               # MFU, HBM SOL%, roofline, FLOPs computation
    config.py                # GLM-5 dims + H100 hardware constants
    report.py                # JSON output + environment snapshot
  moe_sweep/                 # MoE-Inference-Bench style sweeps (SC '25 standard)
    __init__.py
    bench_moe.py             # Batch × tokens × experts × FFN dim sweeps
  triple_report/             # Triple Report: micro → component → end-to-end
    __init__.py
    bench_micro.py           # Kernel-level TFLOPS per component
    bench_component.py       # Full decoder layer integration
    bench_e2e.py             # End-to-end inference (TTFT + TPOT)
  mfu_ceiling/               # MFU relative to FA3's 75% ceiling
    __init__.py
    bench_mfu.py             # MFU at various (B, T, precision) configs
  fp8_pareto/                # FP8 speed-quality Pareto frontier
    __init__.py
    bench_fp8.py             # TFLOPS + cosine similarity at each precision
  run_all.py                 # Orchestrator: run all experiments
```

## Quick Start

```bash
# Run MoE sweeps only (~1 hour):
python -m benchmark.moe_sweep.bench_moe

# Run triple report (~30 min):
python -m benchmark.triple_report.bench_micro
python -m benchmark.triple_report.bench_component
python -m benchmark.triple_report.bench_e2e

# Run MFU ceiling analysis (~20 min):
python -m benchmark.mfu_ceiling.bench_mfu

# Run FP8 Pareto frontier (~30 min):
python -m benchmark.fp8_pareto.bench_fp8

# Run everything (~3 hours):
python -m benchmark.run_all --output-dir results/
```

## Methodology

Aligned with:
- **MoE-Inference-Bench (SC '25)**: batch {1,16,32,64}, tokens {128,256,512,1024,2048}
- **FlashAttention-3 (NeurIPS 2024)**: MFU as % of peak, 75% reference ceiling
- **Sarathi-Serve (OSDI '24)**: TTFT, TPOT, goodput under SLA
- **MLPerf v5.1**: p99 TTFT < 2s, p99 TPOT < 80ms

Statistical: 10 warmup, 100 iterations, bootstrap 95% CI, Mann-Whitney U for comparisons.

---

## DeepGEMM FP8 Scale Tensor Layout — Critical Documentation

**Date discovered:** March 2026, during H100 benchmarking on RunPod (PyTorch 2.8.0+cu128, DeepGEMM latest)

### The Problem

DeepGEMM's FP8 grouped GEMM (`m_grouped_fp8_gemm_nt_contiguous`) requires scale factor tensors with a **specific TMA-aligned 2D layout**. Using the wrong scale shape causes a C++ assertion failure:

```
Assertion error (csrc/apis/../jit_kernels/impls/../heuristics/../../utils/layout.hpp:94):
  sf.dim() == static_cast<int>(num_groups.has_value()) + 2
```

And if dimensions are 2D but wrong size:
```
Assertion error (layout.hpp:97): sf.size(-2) == ceil_div(mn, gran_mn)
```

### Root Cause

DeepGEMM uses **128-element block-wise quantization** with TMA (Tensor Memory Accelerator) aligned scale tensors. The scales are NOT simple per-row or per-tensor values — they follow a specific memory layout that the H100's TMA hardware can efficiently load.

The scale tensor must be shaped as:
```
A scales: [M, ceil(K / 128)]     — 2D, one scale per 128-element block along K
B scales: [E, N, ceil(K / 128)]  — 3D for grouped GEMM (E = num expert groups)
```

Where `128` is the block granularity (`get_mk_alignment_for_contiguous_layout() = 128`).

### What DOES NOT Work

```python
# WRONG: per_custom_dims_cast_to_fp8 with dim (0,) gives 1D scales
a_fp8 = per_custom_dims_cast_to_fp8(a, (0,), False)
# Returns: (tensor[M, K], scales[M])  ← 1D scales, FAILS assertion

# WRONG: Manual 1D or wrong-shape 2D scales
scales = torch.ones(M, dtype=torch.float32)          # 1D — FAILS
scales = torch.ones(M, 1, dtype=torch.float32)        # 2D but wrong size — FAILS
scales = torch.ones(M, K // 128, dtype=torch.float32) # 2D right size but NOT TMA-aligned — may FAIL

# WRONG: per_custom_dims_cast_to_fp8 with ANY dim spec
# (0,) → 1D, (1,) → 1D, (0,1) → 0D scalar — ALL fail the dim assertion
```

### What DOES Work

```python
# CORRECT: Use per_block_cast_to_fp8 — produces TMA-aligned 2D scales
from deep_gemm.utils import per_block_cast_to_fp8

a_fp8 = per_block_cast_to_fp8(a_bf16)
# Returns: (tensor[M, K] as FP8, scales[M, ceil(K/128)] as FP32) — TMA-aligned 2D

# For grouped GEMM B weights [E, N, K]:
b_flat = b_bf16.reshape(E * N, K)
b_fp8_flat = per_block_cast_to_fp8(b_flat)
b_fp8 = (b_fp8_flat[0].view(E, N, K), b_fp8_flat[1].view(E, N, -1))
# Returns: (tensor[E, N, K] as FP8, scales[E, N, ceil(K/128)] as FP32)
```

If additional TMA alignment is needed:
```python
from deep_gemm.utils import get_mn_major_tma_aligned_tensor

# Transform scales to exact TMA-aligned layout
b_scales_aligned = get_mn_major_tma_aligned_tensor(b_fp8[1])
```

### BF16 Grouped GEMM — Always-Working Fallback

BF16 grouped GEMM has NO scale tensors and always works:
```python
# No quantization needed — direct BF16 tensors
deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a_bf16, b_bf16_grouped, output, grouped_layout)
```

This is confirmed working on H100 with PyTorch 2.8.0+cu128. Use this as the baseline when FP8 has issues.

### DeepGEMM Quantization Function Reference

| Function | Input | Output Scales Shape | Use Case |
|----------|-------|-------------------|----------|
| `per_block_cast_to_fp8(x)` | `[M, K]` BF16 | `[M, ceil(K/128)]` FP32 | **Correct for GEMM** — TMA-aligned block scales |
| `per_token_cast_to_fp8(x)` | `[M, K]` BF16 | `[M, 1]` FP32 | Per-token scaling (simpler, less precise) |
| `per_channel_cast_to_fp8(x)` | `[M, K]` BF16 | `[1, K]` FP32 | Per-channel scaling |
| `per_custom_dims_cast_to_fp8(x, (0,))` | `[M, K]` BF16 | `[M]` FP32 ← **1D, FAILS** | **DO NOT USE for GEMM** |

### The `grouped_layout` Tensor

The 4th argument to `m_grouped_fp8_gemm_nt_contiguous` is a **per-row expert index**:
```python
# grouped_layout[i] = expert_id for row i
# Shape: [M], dtype: torch.int32
# Tokens MUST be sorted by expert (contiguous blocks per expert)
grouped_layout = sorted_expert_ids.to(torch.int32)  # [M]
```

It is NOT cumulative expert counts or expert boundaries.

### M-Dimension Alignment

```python
from deep_gemm.utils import get_m_alignment_for_contiguous_layout
alignment = get_m_alignment_for_contiguous_layout()  # Returns 128
# Total M (token count) should ideally be a multiple of 128 for best performance
```

### Version Notes

- Tested on: DeepGEMM (latest as of March 2026), PyTorch 2.8.0+cu128, H100 80GB HBM3
- The `per_custom_dims_cast_to_fp8` function was used in older DeepGEMM examples but produces 1D scales incompatible with current GEMM kernels
- The NVCC 12.9 warning (`Warning: please use at least NVCC 12.9 for the best DeepGEMM performance`) is non-fatal but indicates suboptimal kernel codegen with CUDA 12.8
- DeepGEMM JIT compiles kernels on first call (~10-60s). Cache at `~/.deep_gemm/`
