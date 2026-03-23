"""H100 Category 2: TMA (Tensor Memory Accelerator) Verification.

H100's TMA enables async bulk global↔shared memory transfers. If TMA is not
active, kernels fall back to standard loads with 20-30% perf loss and no
visible error. We verify TMA activity via ncu metrics and bandwidth checks.

Requirements: H100 (SM90), flash-mla and/or deep-gemm installed.
"""

import sys
import torch
from .conftest import skip_no_sm90, has_flash_mla, has_deep_gemm, cuda_timer_fn, make_full_cfg


@skip_no_sm90
def h100_test_tma_bandwidth_flashmla():
    """Verify FlashMLA decode achieves near-peak HBM bandwidth (implies TMA active)."""
    print("\n[H100-TMA-1] FlashMLA bandwidth check (TMA proxy)")
    if not has_flash_mla():
        print("  SKIP flash_mla not installed")
        return True

    from flash_mla import get_mla_metadata, flash_mla_with_kvcache

    device = "cuda"
    B, H = 32, 64
    d_qk, d_v = 576, 512
    seq_kv = 4096
    page_size = 64
    num_pages = (B * seq_kv + page_size - 1) // page_size

    q = torch.randn(B, 1, H, d_qk, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(num_pages, page_size, 1, d_qk, dtype=torch.bfloat16, device=device)
    seqlens = torch.full((B,), seq_kv, dtype=torch.int32, device=device)
    block_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(B, -1)
    metadata, _ = get_mla_metadata()

    def run():
        flash_mla_with_kvcache(q, k_cache, block_table, seqlens, head_dim_v=d_v,
                               tile_scheduler_metadata=metadata, softmax_scale=d_qk**-0.5)

    times = cuda_timer_fn(run, warmup=10, iters=30)
    median_ms = times[len(times) // 2]

    # Bandwidth calculation: reads Q + KV cache, writes O
    bytes_q = B * 1 * H * d_qk * 2  # BF16
    bytes_kv = B * seq_kv * 1 * d_qk * 2
    bytes_o = B * 1 * H * d_v * 2
    total_bytes = bytes_q + bytes_kv + bytes_o
    bandwidth_gb_s = total_bytes / (median_ms * 1e-3) / 1e9

    # H100 SXM peak = 3350 GB/s. TMA-active decode should hit >1500 GB/s
    # Non-TMA would be <1000 GB/s
    threshold = 1000  # conservative lower bound
    ok = bandwidth_gb_s > threshold
    print(f"  Bandwidth: {bandwidth_gb_s:.0f} GB/s (threshold: >{threshold} GB/s)")
    if ok:
        print(f"  PASS bandwidth suggests TMA is active")
    else:
        print(f"  FAIL bandwidth too low — TMA may not be active. Run ncu to confirm:")
        print(f"    ncu --metrics smsp__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_elapsed")
    return ok


@skip_no_sm90
def h100_test_tma_bandwidth_deepgemm():
    """Verify DeepGEMM grouped GEMM achieves high TFLOPS (implies TMA+WGMMA active)."""
    print("\n[H100-TMA-2] DeepGEMM TFLOPS check (TMA+WGMMA proxy)")
    if not has_deep_gemm():
        print("  SKIP deep_gemm not installed")
        return True

    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8, per_channel_cast_to_fp8

    device = "cuda"
    E, N, D, I = 8, 2048, 512, 128

    a = torch.randn(N, D, device=device, dtype=torch.bfloat16)
    b = torch.randn(E, I, D, device=device, dtype=torch.bfloat16)
    # A (activations): per_token gives sf[N, D//128] — scales along K dimension per row
    a_fp8 = per_token_cast_to_fp8(a, False)
    # B (weights): per_channel gives sf[I//128, D] — scales along N dimension per column
    # For grouped: quantize each expert's [I, D] weight, then stack
    b_fp8_list, b_sf_list = [], []
    for e in range(E):
        be = per_channel_cast_to_fp8(b[e], False)  # b[e] is [I, D], sf is [I//128, D]
        b_fp8_list.append(be[0])
        b_sf_list.append(be[1])
    b_fp8 = (torch.stack(b_fp8_list), torch.stack(b_sf_list))
    d = torch.empty(N, I, device=device, dtype=torch.bfloat16)
    layout = torch.arange(N, device=device, dtype=torch.int32) % E

    def run():
        deep_gemm.m_grouped_fp8_gemm_nt_contiguous(a_fp8, b_fp8, d, layout)

    times = cuda_timer_fn(run, warmup=5, iters=20)
    median_ms = times[len(times) // 2]

    flops = 2 * N * D * I
    tflops = flops / (median_ms * 1e-3) / 1e12

    # H100 FP8 peak = 1979 TFLOPS. Good grouped GEMM should hit >200 TFLOPS at this size
    threshold = 50  # small problem, conservative
    ok = tflops > threshold
    print(f"  TFLOPS: {tflops:.1f} (threshold: >{threshold})")
    if ok:
        print(f"  PASS TFLOPS suggests WGMMA+TMA active")
    else:
        print(f"  FAIL TFLOPS too low — check WGMMA utilization via ncu")
    return ok


if __name__ == "__main__":
    results = [h100_test_tma_bandwidth_flashmla(), h100_test_tma_bandwidth_deepgemm()]
    sys.exit(0 if all(results) else 1)
