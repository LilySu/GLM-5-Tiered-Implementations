"""Debug script v2: fix scale tensor dimensions for grouped GEMM."""

import torch
import deep_gemm
from deep_gemm.utils import per_custom_dims_cast_to_fp8

print("=== DeepGEMM Grouped GEMM Debug v2 ===")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

E, I, D, M = 4, 64, 128, 32

a = torch.randn(M, D, dtype=torch.bfloat16, device="cuda")
b = torch.randn(E, I, D, dtype=torch.bfloat16, device="cuda")

# The key insight from the assertion:
# sf.dim() == static_cast<int>(num_groups.has_value()) + 2
# For grouped GEMM, num_groups IS present, so sf.dim() must be 0 + 2 = 2
# But per_custom_dims_cast_to_fp8(a, (0,), False) gives scales with dim=1 [M]
# We need scales with dim=2

# Try 1: Quantize with (0,1) dims to get 2D scales? No — that changes the quantization.
# Try 2: Use get_col_major_tma_aligned_tensor for proper layout
# Try 3: Manually construct the (tensor, scales) with correct dims

# Let's first check what the non-grouped API looks like vs grouped
print("--- Test 1: Non-grouped fp8_gemm_nt (should work) ---")
try:
    a_fp8 = per_custom_dims_cast_to_fp8(a, (0,), False)
    b_single = b[0]  # [I, D] — single expert
    b_fp8 = per_custom_dims_cast_to_fp8(b_single, (0,), False)
    d_out = torch.empty(M, I, dtype=torch.bfloat16, device="cuda")
    deep_gemm.fp8_gemm_nt(a_fp8, b_fp8, d_out)
    print(f"  SUCCESS: a_scales={a_fp8[1].shape} b_scales={b_fp8[1].shape}")
except Exception as e:
    print(f"  FAILED: {e}")

# Check what get_mk_alignment returns
print()
print("--- Test 2: Check alignment requirements ---")
try:
    from deep_gemm.utils import get_mk_alignment_for_contiguous_layout
    alignment = get_mk_alignment_for_contiguous_layout()
    print(f"  MK alignment: {alignment}")
except Exception as e:
    print(f"  No alignment util: {e}")

# Try the m_grouped with BF16 instead (no FP8 scales issue)
print()
print("--- Test 3: m_grouped_bf16_gemm_nt_contiguous (no scales) ---")
try:
    d_out = torch.empty(M, I, dtype=torch.bfloat16, device="cuda")
    grouped_layout = torch.zeros(M, dtype=torch.int32, device="cuda")
    for i in range(M):
        grouped_layout[i] = i % E
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d_out, grouped_layout)
    print(f"  SUCCESS: BF16 grouped GEMM works!")
    print(f"  Output sample: {d_out[0, :4].tolist()}")
except Exception as e:
    print(f"  FAILED: {e}")

# Try FP8 grouped with manually constructed 2D scales
print()
print("--- Test 4: FP8 grouped with 2D scales (manual construction) ---")
try:
    # A: quantize and make scales 2D
    a_fp8_tensor = a.to(torch.float8_e4m3fn)
    a_scales = torch.ones(M, D // 128, dtype=torch.float32, device="cuda")  # per-128-element blocks
    print(f"  a_fp8: {a_fp8_tensor.shape}, a_scales: {a_scales.shape}")

    # B: already 3D with 2D scales
    b_fp8_tensor = b.to(torch.float8_e4m3fn)
    b_scales = torch.ones(E, I, dtype=torch.float32, device="cuda")
    print(f"  b_fp8: {b_fp8_tensor.shape}, b_scales: {b_scales.shape}")

    d_out = torch.empty(M, I, dtype=torch.bfloat16, device="cuda")
    grouped_layout = torch.zeros(M, dtype=torch.int32, device="cuda")
    for i in range(M):
        grouped_layout[i] = i % E

    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        (a_fp8_tensor, a_scales), (b_fp8_tensor, b_scales), d_out, grouped_layout
    )
    print(f"  SUCCESS: FP8 grouped GEMM with 2D scales works!")
except Exception as e:
    print(f"  FAILED: {e}")

# Try with different A scale shapes
print()
print("--- Test 5: Sweep A scale dimensions ---")
for a_scale_shape in [(M,), (M, 1), (M, D // 128)]:
    try:
        a_fp8_tensor = a.to(torch.float8_e4m3fn)
        a_scales = torch.ones(a_scale_shape, dtype=torch.float32, device="cuda")
        b_fp8_tensor = b.to(torch.float8_e4m3fn)
        b_scales = torch.ones(E, I, dtype=torch.float32, device="cuda")
        d_out = torch.empty(M, I, dtype=torch.bfloat16, device="cuda")
        grouped_layout = torch.zeros(M, dtype=torch.int32, device="cuda")
        for i in range(M):
            grouped_layout[i] = i % E
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_fp8_tensor, a_scales), (b_fp8_tensor, b_scales), d_out, grouped_layout
        )
        print(f"  a_scales={a_scale_shape}: SUCCESS")
    except Exception as e:
        print(f"  a_scales={a_scale_shape}: FAILED — {str(e)[:80]}")

# Try with per_custom_dims but reshape scales to 2D
print()
print("--- Test 6: per_custom_dims_cast_to_fp8 with reshaped scales ---")
try:
    a_fp8_raw = per_custom_dims_cast_to_fp8(a, (0,), False)
    # Reshape 1D scales [M] to 2D [M, 1]
    a_fp8_2d = (a_fp8_raw[0], a_fp8_raw[1].unsqueeze(-1))

    b_fp8_raw = per_custom_dims_cast_to_fp8(b.view(E * I, D), (0,), False)
    b_fp8_3d = (b_fp8_raw[0].view(E, I, D), b_fp8_raw[1].view(E, I))

    d_out = torch.empty(M, I, dtype=torch.bfloat16, device="cuda")
    grouped_layout = torch.zeros(M, dtype=torch.int32, device="cuda")
    for i in range(M):
        grouped_layout[i] = i % E

    print(f"  a: tensor={a_fp8_2d[0].shape} scales={a_fp8_2d[1].shape}")
    print(f"  b: tensor={b_fp8_3d[0].shape} scales={b_fp8_3d[1].shape}")

    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
        a_fp8_2d, b_fp8_3d, d_out, grouped_layout
    )
    print(f"  SUCCESS!")
except Exception as e:
    print(f"  FAILED: {str(e)[:120]}")

print()
print("=== Done ===")
