"""Debug v5: per_block_cast_to_fp8 requires use_ue8m0 argument."""

import torch
import inspect
import deep_gemm
from deep_gemm.utils import (
    per_block_cast_to_fp8,
    per_token_cast_to_fp8,
    get_tma_aligned_size,
    get_m_alignment_for_contiguous_layout,
)

print("=== DeepGEMM FP8 — per_block_cast_to_fp8 signature ===")

# Check the exact signature
sig = inspect.signature(per_block_cast_to_fp8)
print(f"per_block_cast_to_fp8 signature: {sig}")
print()

# Also check per_token_cast_to_fp8
sig2 = inspect.signature(per_token_cast_to_fp8)
print(f"per_token_cast_to_fp8 signature: {sig2}")
print()

# Check all cast functions
for name in ['per_block_cast_to_fp8', 'per_token_cast_to_fp8', 'per_channel_cast_to_fp8', 'per_custom_dims_cast_to_fp8']:
    fn = getattr(deep_gemm.utils, name, None)
    if fn:
        print(f"{name}: {inspect.signature(fn)}")
print()

M, N, K = 32, 64, 128
E = 4

a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")

# Try per_block_cast_to_fp8 with use_ue8m0=True and False
print("--- per_block_cast_to_fp8(x, use_ue8m0=True) ---")
try:
    result = per_block_cast_to_fp8(a_bf16, True)
    print(f"  tensor={result[0].shape} dtype={result[0].dtype}")
    print(f"  scales={result[1].shape} dtype={result[1].dtype}")
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("--- per_block_cast_to_fp8(x, use_ue8m0=False) ---")
try:
    result = per_block_cast_to_fp8(a_bf16, False)
    print(f"  tensor={result[0].shape} dtype={result[0].dtype}")
    print(f"  scales={result[1].shape} dtype={result[1].dtype}")
except Exception as e:
    print(f"  FAILED: {e}")

# Test non-grouped fp8_gemm_nt with per_block scales
print()
print("--- fp8_gemm_nt with per_block_cast_to_fp8 ---")
b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")
for ue8m0 in [True, False]:
    try:
        a_fp8 = per_block_cast_to_fp8(a_bf16, ue8m0)
        b_fp8 = per_block_cast_to_fp8(b_bf16, ue8m0)
        d_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d_out)
        print(f"  ue8m0={ue8m0}: SUCCESS (a_scales={a_fp8[1].shape}, b_scales={b_fp8[1].shape})")
    except Exception as e:
        print(f"  ue8m0={ue8m0}: FAILED — {str(e)[:100]}")

# Test grouped fp8 GEMM
print()
print("--- m_grouped_fp8_gemm_nt_contiguous with per_block_cast_to_fp8 ---")
b_grouped_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
for ue8m0 in [True, False]:
    try:
        a_fp8 = per_block_cast_to_fp8(a_bf16, ue8m0)
        b_flat = b_grouped_bf16.reshape(E * N, K)
        b_fp8_flat = per_block_cast_to_fp8(b_flat, ue8m0)
        b_fp8 = (b_fp8_flat[0].view(E, N, K), b_fp8_flat[1].view(E, N, -1))

        d_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
        grouped_layout = torch.zeros(M, dtype=torch.int32, device="cuda")
        for i in range(M):
            grouped_layout[i] = i % E

        print(f"  ue8m0={ue8m0}: a=({a_fp8[0].shape}, {a_fp8[1].shape}) b=({b_fp8[0].shape}, {b_fp8[1].shape})")
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a_fp8, b_fp8, d_out, grouped_layout)
        print(f"  ue8m0={ue8m0}: SUCCESS!")
    except Exception as e:
        print(f"  ue8m0={ue8m0}: FAILED — {str(e)[:120]}")

# Also try with get_mn_major_tma_aligned_packed_ue8m0_tensor
print()
print("--- With TMA-aligned packed UE8M0 scale transform ---")
try:
    from deep_gemm.utils import get_mn_major_tma_aligned_packed_ue8m0_tensor
    a_fp8 = per_block_cast_to_fp8(a_bf16, True)
    print(f"  Raw a scales: {a_fp8[1].shape} dtype={a_fp8[1].dtype}")
    a_scales_tma = get_mn_major_tma_aligned_packed_ue8m0_tensor(a_fp8[1])
    print(f"  TMA-aligned a scales: {a_scales_tma.shape} dtype={a_scales_tma.dtype}")

    b_flat = b_grouped_bf16.reshape(E * N, K)
    b_fp8_flat = per_block_cast_to_fp8(b_flat, True)
    b_scales_3d = b_fp8_flat[1].view(E, N, -1)
    b_scales_tma = get_mn_major_tma_aligned_packed_ue8m0_tensor(b_scales_3d)
    print(f"  TMA-aligned b scales: {b_scales_tma.shape} dtype={b_scales_tma.dtype}")

    d_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (a_fp8[0], a_scales_tma),
        (b_fp8_flat[0].view(E, N, K), b_scales_tma),
        d_out, grouped_layout
    )
    print(f"  SUCCESS with TMA-aligned packed UE8M0!")
except Exception as e:
    print(f"  FAILED: {str(e)[:150]}")

# Full GLM-5 dims BF16 benchmark (always works)
print()
print("--- BF16 grouped GEMM at GLM-5 dims (fallback benchmark) ---")
try:
    E_full, I_full, D_full = 256, 2048, 6144
    M_full = 256 * 32

    a_full = torch.randn(M_full, D_full, dtype=torch.bfloat16, device="cuda")
    b_full = torch.randn(E_full, I_full, D_full, dtype=torch.bfloat16, device="cuda")
    d_full = torch.empty(M_full, I_full, dtype=torch.bfloat16, device="cuda")
    layout_full = torch.arange(M_full, device="cuda", dtype=torch.int32) % E_full

    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a_full, b_full, d_full, layout_full)
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(10):
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a_full, b_full, d_full, layout_full)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 10
    flops = 2 * M_full * I_full * D_full
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"  {ms:.2f} ms, {tflops:.1f} TFLOPS (BF16 peak: 989 TFLOPS, MFU: {tflops/989*100:.1f}%)")
except torch.cuda.OutOfMemoryError:
    print(f"  OOM")
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("=== Done ===")
