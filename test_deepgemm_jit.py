import torch, deep_gemm
a = torch.randn(32, 128, dtype=torch.bfloat16, device='cuda')
b = torch.randn(1, 64, 128, dtype=torch.bfloat16, device='cuda')
d = torch.empty(32, 64, dtype=torch.bfloat16, device='cuda')
layout = torch.zeros(32, dtype=torch.int32, device='cuda')
deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, layout)
print('DeepGEMM JIT works!')
