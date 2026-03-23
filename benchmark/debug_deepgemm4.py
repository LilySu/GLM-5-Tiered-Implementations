"""Debug v4: Use per_block_cast_to_fp8 and TMA-aligned scale tensors."""

import torch
import deep_gemm
from deep_gemm.utils import (
    per_block_cast_to_fp8,
    per_token_cast_to_fp8,
    get_mn_major_tma_aligned_tensor,
    get_tma_aligned_size,
    get_m_alignment_for_contiguous_layout,
)

print("=== DeepGEMM FP8 — per_block_cast_to_fp8 ===")

M, N, K = 32, 64, 128
E = 4

a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

# Test per_block_cast_to_fp8
print("--- per_block_cast_to_fp8 ---")
a_fp8 = per_block_cast_to_fp8(a_bf16)
print(f"  a: tensor={a_fp8[0].shape} dtype={a_fp8[0].dtype}")
print(f"     scales={a_fp8[1].shape} dtype={a_fp8[1].dtype}")

b_fp8 = per_block_cast_to_fp8(b_bf16)
print(f"  b: tensor={b_fp8[0].shape} dtype={b_fp8[0].dtype}")
print(f"     scales={b_fp8[1].shape} dtype={b_fp8[1].dtype}")

# Test per_token_cast_to_fp8
print()
print("--- per_token_cast_to_fp8 ---")
a_tok = per_token_cast_to_fp8(a_bf16)
print(f"  a: tensor={a_tok[0].shape} scales={a_tok[1].shape}")

# Test non-grouped fp8_gemm_nt with per_block scales
print()
print("--- fp8_gemm_nt with per_block_cast_to_fp8 ---")
try:
    d_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d_out)
    print(f"  SUCCESS! Output: {d_out[0, :4].tolist()}")
except Exception as e:
    print(f"  FAILED: {str(e)[:120]}")

# Now test grouped with per_block scales
print()
print("--- m_grouped_fp8_gemm_nt_contiguous with per_block scales ---")
try:
    b_grouped_bf16 = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")

    # Quantize A with per_block
    a_fp8 = per_block_cast_to_fp8(a_bf16)
    print(f"  a: tensor={a_fp8[0].shape} scales={a_fp8[1].shape}")

    # Quantize B: need [E, N, K] with scales [E, N, K//128] or similar
    # Flatten to [E*N, K], quantize, reshape back
    b_flat = b_grouped_bf16.reshape(E * N, K)
    b_fp8_flat = per_block_cast_to_fp8(b_flat)
    print(f"  b flat: tensor={b_fp8_flat[0].shape} scales={b_fp8_flat[1].shape}")

    b_fp8 = (b_fp8_flat[0].view(E, N, K), b_fp8_flat[1].view(E, N, -1))
    print(f"  b grouped: tensor={b_fp8[0].shape} scales={b_fp8[1].shape}")

    d_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    grouped_layout = torch.zeros(M, dtype=torch.int32, device="cuda")
    for i in range(M):
        grouped_layout[i] = i % E

    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a_fp8, b_fp8, d_out, grouped_layout)
    print(f"  SUCCESS! Output: {d_out[0, :4].tolist()}")
except Exception as e:
    print(f"  FAILED: {str(e)[:150]}")

# Try with TMA-aligned scale transform
print()
print("--- With get_mn_major_tma_aligned_tensor for scales ---")
try:
    b_flat = b_grouped_bf16.reshape(E * N, K)
    b_fp8_flat = per_block_cast_to_fp8(b_flat)
    print(f"  b flat scales raw: {b_fp8_flat[1].shape}")

    # Transform scales to TMA-aligned layout
    b_scales_aligned = get_mn_major_tma_aligned_tensor(b_fp8_flat[1].view(E, N, -1))
    print(f"  b scales TMA-aligned: {b_scales_aligned.shape}")

    b_fp8 = (b_fp8_flat[0].view(E, N, K), b_scales_aligned)

    # Also align A scales
    a_scales_aligned = get_mn_major_tma_aligned_tensor(a_fp8[1])
    print(f"  a scales TMA-aligned: {a_scales_aligned.shape}")

    d_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (a_fp8[0], a_scales_aligned), b_fp8, d_out, grouped_layout
    )
    print(f"  SUCCESS! Output: {d_out[0, :4].tolist()}")
except Exception as e:
    print(f"  FAILED: {str(e)[:150]}")

# Full GLM-5 dims test
print()
print("--- Full GLM-5 dims: E=256, I=2048, D=6144, M=8192 (256*32) ---")
try:
    E_full, I_full, D_full = 256, 2048, 6144
    M_full = 256 * 32  # 32 tokens per expert avg

    m_align = get_m_alignment_for_contiguous_layout()
    print(f"  M alignment: {m_align}")

    # Just test BF16 grouped at full dims (FP8 can wait)
    a_full = torch.randn(M_full, D_full, dtype=torch.bfloat16, device="cuda")
    b_full = torch.randn(E_full, I_full, D_full, dtype=torch.bfloat16, device="cuda")
    d_full = torch.empty(M_full, I_full, dtype=torch.bfloat16, device="cuda")
    layout_full = torch.zeros(M_full, dtype=torch.int32, device="cuda")
    for i in range(M_full):
        layout_full[i] = i % E_full

    # Warmup
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
    print(f"  BF16 grouped GEMM at GLM-5 dims: {ms:.2f} ms, {tflops:.1f} TFLOPS")
except torch.cuda.OutOfMemoryError:
    print(f"  OOM at full dims")
except Exception as e:
    print(f"  FAILED: {str(e)[:150]}")

print()
print("=== Done ===")
