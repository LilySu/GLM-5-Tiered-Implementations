"""Debug v3: Find the exact scale tensor shape DeepGEMM FP8 wants."""

import torch
import deep_gemm

print("=== DeepGEMM FP8 Scale Shape Debug ===")
print(f"DeepGEMM version functions: {[x for x in dir(deep_gemm) if not x.startswith('_')]}")
print()

# Check if there's a get_tma_aligned_size or similar utility
from deep_gemm import utils as dg_utils
print(f"deep_gemm.utils contents: {[x for x in dir(dg_utils) if not x.startswith('_')]}")
print()

M, N, K = 32, 64, 128
E = 4

a_bf16 = torch.randn(M, K, dtype=torch.bfloat16, device="cuda")
b_bf16 = torch.randn(N, K, dtype=torch.bfloat16, device="cuda")

# Test per_custom_dims_cast_to_fp8 with different dim specs
print("--- per_custom_dims_cast_to_fp8 output shapes ---")
for dims in [(0,), (1,), (0, 1)]:
    try:
        result = dg_utils.per_custom_dims_cast_to_fp8(a_bf16, dims, False)
        print(f"  dims={dims}: tensor={result[0].shape} scales={result[1].shape}")
    except Exception as e:
        print(f"  dims={dims}: FAILED — {str(e)[:80]}")

# The key: try (1,) which gives per-column-block scales → 2D
print()
print("--- Test: fp8_gemm_nt with (1,) quantized A ---")
try:
    a_fp8 = dg_utils.per_custom_dims_cast_to_fp8(a_bf16, (1,), False)
    b_fp8 = dg_utils.per_custom_dims_cast_to_fp8(b_bf16, (1,), False)
    print(f"  a: tensor={a_fp8[0].shape} scales={a_fp8[1].shape}")
    print(f"  b: tensor={b_fp8[0].shape} scales={b_fp8[1].shape}")
    d_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d_out)
    print(f"  SUCCESS!")
except Exception as e:
    print(f"  FAILED: {str(e)[:120]}")

# Try (0,1) which gives scalar scale
print()
print("--- Test: fp8_gemm_nt with (0,1) quantized ---")
try:
    a_fp8 = dg_utils.per_custom_dims_cast_to_fp8(a_bf16, (0, 1), False)
    b_fp8 = dg_utils.per_custom_dims_cast_to_fp8(b_bf16, (0, 1), False)
    print(f"  a: tensor={a_fp8[0].shape} scales={a_fp8[1].shape}")
    print(f"  b: tensor={b_fp8[0].shape} scales={b_fp8[1].shape}")
    d_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d_out)
    print(f"  SUCCESS!")
except Exception as e:
    print(f"  FAILED: {str(e)[:120]}")

# Check if there's a cell-level cast function
print()
print("--- Looking for cell/block quantization ---")
for name in dir(dg_utils):
    if "cast" in name.lower() or "quant" in name.lower() or "fp8" in name.lower():
        fn = getattr(dg_utils, name)
        print(f"  {name}: {fn}")
        if hasattr(fn, "__doc__") and fn.__doc__:
            print(f"    doc: {fn.__doc__[:200]}")

# Manual block-wise quantization matching DeepGEMM's expected layout
print()
print("--- Test: Manual 128-element block quantization with 2D scales ---")
BLOCK = 128
try:
    # A: [M, K] → quantize per 128-element row blocks → scales [M, K//128]
    num_blocks_k = (K + BLOCK - 1) // BLOCK
    a_fp8_tensor = a_bf16.to(torch.float8_e4m3fn)
    a_blocked = a_bf16.view(M, num_blocks_k, BLOCK)
    a_scales = a_blocked.abs().amax(dim=-1) / 448.0  # [M, num_blocks_k]
    a_scales = a_scales.to(torch.float32)
    print(f"  a_fp8: {a_fp8_tensor.shape}, a_scales: {a_scales.shape} (2D: {a_scales.dim()}D)")

    b_3d = b_bf16.unsqueeze(0).expand(E, N, K)  # [E, N, K]
    b_fp8_tensor = b_3d.to(torch.float8_e4m3fn)
    b_blocked = b_3d.reshape(E, N, num_blocks_k, BLOCK)
    b_scales = b_blocked.abs().amax(dim=-1) / 448.0  # [E, N, num_blocks_k]
    b_scales = b_scales.to(torch.float32)
    print(f"  b_fp8: {b_fp8_tensor.shape}, b_scales: {b_scales.shape} (3D: {b_scales.dim()}D)")

    d_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    grouped_layout = torch.zeros(M, dtype=torch.int32, device="cuda")
    for i in range(M):
        grouped_layout[i] = i % E

    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (a_fp8_tensor, a_scales), (b_fp8_tensor, b_scales), d_out, grouped_layout
    )
    print(f"  SUCCESS with manual block scales!")
    print(f"  Output sample: {d_out[0, :4].tolist()}")
except Exception as e:
    print(f"  FAILED: {str(e)[:150]}")

# Also try the BF16 grouped path as our fallback
print()
print("--- Fallback: BF16 grouped GEMM benchmark (no FP8) ---")
try:
    b_grouped = torch.randn(E, N, K, dtype=torch.bfloat16, device="cuda")
    d_out = torch.empty(M, N, dtype=torch.bfloat16, device="cuda")
    grouped_layout = torch.zeros(M, dtype=torch.int32, device="cuda")
    for i in range(M):
        grouped_layout[i] = i % E

    # Warmup
    for _ in range(3):
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a_bf16, b_grouped, d_out, grouped_layout)
    torch.cuda.synchronize()

    # Time it
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(100):
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a_bf16, b_grouped, d_out, grouped_layout)
    end.record()
    torch.cuda.synchronize()
    ms = start.elapsed_time(end) / 100
    flops = 2 * M * N * K
    tflops = flops / (ms * 1e-3) / 1e12
    print(f"  BF16 grouped GEMM: {ms:.3f} ms, {tflops:.2f} TFLOPS")
except Exception as e:
    print(f"  FAILED: {e}")

print()
print("=== Done ===")
