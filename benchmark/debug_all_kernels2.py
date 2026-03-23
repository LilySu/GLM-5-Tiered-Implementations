"""Debug v2: Skip inspect.signature for built-in methods, test calls directly."""

import torch

print("=" * 70)
print("  Kernel API Diagnostic v2 — H100")
print("=" * 70)
print(f"PyTorch: {torch.__version__}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print()

# ════════════════════════════════════════════════════════════════════════
# 1. FlashMLA — CONFIRMED WORKING
# ════════════════════════════════════════════════════════════════════════
print("1. FlashMLA dense decode: CONFIRMED WORKING (previous run)")
print("   q=[B,1,64,576] k_cache=[pages,64,1,576] -> out=[B,1,64,512]")
print()

# ════════════════════════════════════════════════════════════════════════
# 2. DeepGEMM fp8_mqa_logits — test all approaches
# ════════════════════════════════════════════════════════════════════════
print("=" * 70)
print("  2. DeepGEMM fp8_mqa_logits")
print("=" * 70)

import deep_gemm
from deep_gemm.utils import per_block_cast_to_fp8, per_token_cast_to_fp8

# Print docstring instead of signature (PyCapsule doesn't support inspect)
doc = getattr(deep_gemm.fp8_mqa_logits, '__doc__', None)
if doc:
    print(f"Docstring:\n{doc[:500]}")
else:
    print("No docstring available")
print()

seq_len = 1
seq_len_kv = 256
num_heads = 32
head_dim = 128

q_bf16 = torch.randn(seq_len, num_heads, head_dim, dtype=torch.bfloat16, device="cuda")
kv_bf16 = torch.randn(seq_len_kv, head_dim, dtype=torch.bfloat16, device="cuda")
weights = torch.randn(seq_len, num_heads, dtype=torch.float32, device="cuda")
cu_k_start = torch.zeros(seq_len, dtype=torch.int32, device="cuda")
cu_k_end = torch.full((seq_len,), seq_len_kv, dtype=torch.int32, device="cuda")

approaches = {}

# Approach A: raw FP8 q (no tuple) + per_token kv tuple
print("--- A: raw FP8 q + per_token kv ---")
try:
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    kv_tok = per_token_cast_to_fp8(kv_bf16, True)
    print(f"  q={q_fp8.shape} dtype={q_fp8.dtype}")
    print(f"  kv=({kv_tok[0].shape}, {kv_tok[1].shape})")
    print(f"  w={weights.shape} cu_start={cu_k_start.shape} cu_end={cu_k_end.shape}")
    deep_gemm.fp8_mqa_logits(q_fp8, kv_tok, weights, cu_k_start, cu_k_end)
    print("  SUCCESS!")
    approaches['A'] = True
except Exception as e:
    print(f"  FAILED: {str(e)[:150]}")
    approaches['A'] = False

# Approach B: per_token q tuple + per_token kv tuple
print("\n--- B: per_token q tuple + per_token kv tuple ---")
try:
    q_flat = q_bf16.reshape(seq_len * num_heads, head_dim)
    q_tok = per_token_cast_to_fp8(q_flat, True)
    q_tok_3d = (q_tok[0].view(seq_len, num_heads, head_dim), q_tok[1].view(seq_len, num_heads))
    kv_tok = per_token_cast_to_fp8(kv_bf16, True)
    print(f"  q=({q_tok_3d[0].shape}, {q_tok_3d[1].shape})")
    print(f"  kv=({kv_tok[0].shape}, {kv_tok[1].shape})")
    deep_gemm.fp8_mqa_logits(q_tok_3d, kv_tok, weights, cu_k_start, cu_k_end)
    print("  SUCCESS!")
    approaches['B'] = True
except Exception as e:
    print(f"  FAILED: {str(e)[:150]}")
    approaches['B'] = False

# Approach C: per_block q tuple + per_block kv tuple
print("\n--- C: per_block q tuple + per_block kv tuple ---")
try:
    q_flat = q_bf16.reshape(seq_len * num_heads, head_dim)
    q_blk = per_block_cast_to_fp8(q_flat, True)
    q_blk_3d = (q_blk[0].view(seq_len, num_heads, head_dim), q_blk[1])
    kv_blk = per_block_cast_to_fp8(kv_bf16, True)
    print(f"  q=({q_blk_3d[0].shape}, {q_blk_3d[1].shape})")
    print(f"  kv=({kv_blk[0].shape}, {kv_blk[1].shape})")
    deep_gemm.fp8_mqa_logits(q_blk_3d, kv_blk, weights, cu_k_start, cu_k_end)
    print("  SUCCESS!")
    approaches['C'] = True
except Exception as e:
    print(f"  FAILED: {str(e)[:150]}")
    approaches['C'] = False

# Approach D: raw FP8 q + raw FP8 kv with separate scale
print("\n--- D: raw FP8 q + (raw FP8 kv, 1D scales) ---")
try:
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    kv_fp8 = kv_bf16.to(torch.float8_e4m3fn)
    kv_scales = torch.ones(seq_len_kv, dtype=torch.float32, device="cuda")
    print(f"  q={q_fp8.shape}")
    print(f"  kv=({kv_fp8.shape}, {kv_scales.shape})")
    deep_gemm.fp8_mqa_logits(q_fp8, (kv_fp8, kv_scales), weights, cu_k_start, cu_k_end)
    print("  SUCCESS!")
    approaches['D'] = True
except Exception as e:
    print(f"  FAILED: {str(e)[:150]}")
    approaches['D'] = False

# Approach E: per_token q (keep 3D) + per_token kv
print("\n--- E: per_token on 3D q directly + per_token kv ---")
try:
    # Quantize q as 3D tensor directly
    q_3d = q_bf16  # [1, 32, 128]
    q_fp8_3d = q_3d.to(torch.float8_e4m3fn)
    q_scales = q_3d.abs().amax(dim=-1, keepdim=False).float() / 448.0  # [1, 32]

    kv_tok = per_token_cast_to_fp8(kv_bf16, True)
    print(f"  q=({q_fp8_3d.shape}, {q_scales.shape})")
    print(f"  kv=({kv_tok[0].shape}, {kv_tok[1].shape})")
    deep_gemm.fp8_mqa_logits((q_fp8_3d, q_scales), kv_tok, weights, cu_k_start, cu_k_end)
    print("  SUCCESS!")
    approaches['E'] = True
except Exception as e:
    print(f"  FAILED: {str(e)[:150]}")
    approaches['E'] = False

# Approach F: Check if there's a clean_logits param
print("\n--- F: with clean_logits=True ---")
try:
    q_fp8 = q_bf16.to(torch.float8_e4m3fn)
    kv_tok = per_token_cast_to_fp8(kv_bf16, True)
    deep_gemm.fp8_mqa_logits(q_fp8, kv_tok, weights, cu_k_start, cu_k_end, True)
    print("  SUCCESS with clean_logits!")
    approaches['F'] = True
except Exception as e:
    print(f"  FAILED: {str(e)[:150]}")
    approaches['F'] = False

print()
print("Summary:")
for k, v in approaches.items():
    print(f"  Approach {k}: {'SUCCESS' if v else 'FAILED'}")

# ════════════════════════════════════════════════════════════════════════
# 3. BF16 grouped GEMM — benchmark
# ════════════════════════════════════════════════════════════════════════
print()
print("=" * 70)
print("  3. BF16 grouped GEMM benchmark")
print("=" * 70)

E, I, D, M = 256, 2048, 6144, 8192
a = torch.randn(M, D, dtype=torch.bfloat16, device="cuda")
b = torch.randn(E, I, D, dtype=torch.bfloat16, device="cuda")
d = torch.empty(M, I, dtype=torch.bfloat16, device="cuda")
layout = (torch.arange(M, device="cuda", dtype=torch.int32) % E)

deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, layout)
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(20):
    deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, layout)
end.record()
torch.cuda.synchronize()
ms = start.elapsed_time(end) / 20
flops = 2 * M * I * D
tflops = flops / (ms * 1e-3) / 1e12
print(f"  {ms:.3f} ms, {tflops:.1f} TFLOPS ({tflops/989*100:.1f}% MFU)")

print()
print("=" * 70)
print("  Done")
print("=" * 70)
