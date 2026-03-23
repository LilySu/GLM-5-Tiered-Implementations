"""Debug script: verify DeepGEMM grouped GEMM API works on this GPU."""

import torch
import deep_gemm
from deep_gemm.utils import per_custom_dims_cast_to_fp8

print("=== DeepGEMM API Debug ===")
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"DeepGEMM functions: {[x for x in dir(deep_gemm) if 'gemm' in x.lower()]}")
print()

# Small dims for testing
E, I, D, M = 4, 64, 128, 32

a = torch.randn(M, D, dtype=torch.bfloat16, device="cuda")
b = torch.randn(E, I, D, dtype=torch.bfloat16, device="cuda")

print("--- Quantizing A ---")
a_fp8 = per_custom_dims_cast_to_fp8(a, (0,), False)
print(f"a_fp8: tensor={a_fp8[0].shape} dtype={a_fp8[0].dtype} scales={a_fp8[1].shape} dtype={a_fp8[1].dtype}")

print("--- Quantizing B (flat then reshape) ---")
b_flat = b.view(E * I, D)
b_fp8_flat = per_custom_dims_cast_to_fp8(b_flat, (0,), False)
print(f"b_fp8 flat: tensor={b_fp8_flat[0].shape} scales={b_fp8_flat[1].shape}")

b_fp8 = (b_fp8_flat[0].view(E, I, D), b_fp8_flat[1].view(E, I))
print(f"b_fp8 reshaped: tensor={b_fp8[0].shape} scales={b_fp8[1].shape}")

print("--- Output and layout ---")
d = torch.empty(M, I, dtype=torch.bfloat16, device="cuda")
grouped_layout = torch.zeros(M, dtype=torch.int32, device="cuda")
for i in range(M):
    grouped_layout[i] = i % E
print(f"d: {d.shape}")
print(f"grouped_layout: {grouped_layout.shape} dtype={grouped_layout.dtype}")
print(f"grouped_layout values: {grouped_layout.tolist()}")

print()
print("--- Calling m_grouped_fp8_gemm_nt_contiguous ---")
try:
    deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a_fp8, b_fp8, d, grouped_layout)
    print("SUCCESS! Output shape:", d.shape)
    print("Output sample:", d[0, :4].tolist())
except Exception as e:
    print(f"FAILED: {e}")
    print()
    print("--- Trying alternative: pass num_groups explicitly ---")
    try:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a_fp8, b_fp8, d, grouped_layout, num_groups=E)
        print("SUCCESS with num_groups!")
    except Exception as e2:
        print(f"FAILED with num_groups: {e2}")

    print()
    print("--- Listing all grouped GEMM signatures ---")
    fn = deep_gemm.m_grouped_fp8_gemm_nt_contiguous
    print(f"Function: {fn}")
    if hasattr(fn, "__doc__"):
        print(f"Docstring: {fn.__doc__}")

    print()
    print("--- Trying with different scale shapes ---")
    # Maybe scales need to be 2D for A as well?
    a_scales_2d = a_fp8[1].unsqueeze(-1)
    print(f"a_scales_2d: {a_scales_2d.shape}")
    try:
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(
            (a_fp8[0], a_scales_2d), b_fp8, d, grouped_layout
        )
        print("SUCCESS with 2D a_scales!")
    except Exception as e3:
        print(f"FAILED: {e3}")

print()
print("=== Done ===")
