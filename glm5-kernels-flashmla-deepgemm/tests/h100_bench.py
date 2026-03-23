"""H100 Benchmark & Profiling Harness for GLM-5 CUDA Kernels.

This harness is designed to run on one or more H100 GPUs. It benchmarks each
kernel-accelerated component individually and the full model end-to-end, and
collects ncu/nsys profiling metrics.

Usage:
    # Quick benchmark (no profiling, just timing):
    python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode bench

    # Full nsys profile (generates .nsys-rep file):
    nsys profile -o glm5_profile python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode nsys

    # ncu kernel-level metrics (slow, targets specific kernels):
    ncu --set full --target-processes all -o glm5_ncu \\
        python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode ncu

    # Multi-GPU (tensor parallel across N GPUs):
    torchrun --nproc_per_node=N -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode bench --tp N

Hardware requirements:
    - 1+ NVIDIA H100/H800 GPUs (SM90)
    - CUDA 12.8+
    - flash-mla and deep-gemm installed from source
    - For full 744B model: 8+ H100 80GB (tensor parallel)
    - For single-layer benchmarks: 1 H100 sufficient

NCU Metrics Collected (when --mode ncu):
    See NCU_METRICS dict below for the full list with descriptions.

NSYS Metrics (when --mode nsys):
    nsys captures the full CUDA timeline including kernel launches, memory ops,
    NCCL collisions, and Python overhead. View with: nsys-ui glm5_profile.nsys-rep
"""

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.distributed as dist

# ── ncu metric groups ────────────────────────────────────────────────────

NCU_METRICS = {
    # ── Throughput ──
    "sm__throughput.avg.pct_of_peak_sustained_elapsed":
        "SM utilization (% of peak). Target: >80% for compute-bound kernels.",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed":
        "HBM bandwidth utilization (% of peak 3.35 TB/s). Target: >70% for memory-bound.",
    "l2__throughput.avg.pct_of_peak_sustained_elapsed":
        "L2 cache throughput. High = good cache reuse.",

    # ── Tensor Core utilization ──
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed":
        "Tensor core utilization (HMMA). Target: >60% for GEMM kernels.",
    "smsp__inst_executed_pipe_tensor_op_hmma.avg.pct_of_peak_sustained_elapsed":
        "HMMA instruction throughput.",

    # ── Warp stalls ──
    "smsp__warps_issue_stalled_wait.pct":
        "Warps stalled waiting for data. High = memory-bound or barrier-bound.",
    "smsp__warps_issue_stalled_no_instruction.pct":
        "Warps stalled on instruction fetch. High = instruction cache miss.",
    "smsp__warps_issue_stalled_mio_throttle.pct":
        "Warps stalled on MIO (TMA). High = TMA descriptor bottleneck.",
    "smsp__warps_issue_stalled_math_pipe_throttle.pct":
        "Warps stalled on math pipe. High = compute-bound (good for GEMM).",

    # ── Memory ──
    "dram__bytes_read.sum":
        "Total HBM bytes read. Compare to theoretical minimum.",
    "dram__bytes_write.sum":
        "Total HBM bytes written.",
    "l2__read_transactions.sum":
        "L2 read transactions. Low/dram ratio = poor caching.",
    "lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum":
        "L2 cache hit rate for texture reads (TMA uses this).",

    # ── TMA (Tensor Memory Accelerator) — H100 specific ──
    "smsp__inst_executed_pipe_uniform.avg.pct_of_peak_sustained_elapsed":
        "Uniform datapath (TMA descriptor setup). High = TMA active.",
    "sm__sass_thread_inst_executed_op_generic_ld.sum":
        "Generic loads. Low = most loads go through TMA (good).",

    # ── Instruction mix ──
    "smsp__inst_executed.sum":
        "Total instructions executed.",
    "sm__sass_thread_inst_executed_op_ffma_pred_on.sum":
        "FP32 FMA ops. Should be low (we want tensor core ops).",
    "sm__sass_thread_inst_executed_op_hfma2_pred_on.sum":
        "FP16 FMA ops.",

    # ── Occupancy ──
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed":
        "Active warp occupancy. Target: >50%.",
    "sm__maximum_warps_per_active_cycle_pct":
        "Theoretical max warps that could be active.",

    # ── Kernel timing ──
    "gpu__time_duration.sum":
        "Kernel execution time (ns).",
    "gpu__cycles_elapsed.sum":
        "Kernel execution cycles.",
}

# Subset for quick profiling (fewer metrics = faster ncu)
NCU_METRICS_QUICK = [
    "sm__throughput.avg.pct_of_peak_sustained_elapsed",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed",
    "sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed",
    "smsp__warps_issue_stalled_wait.pct",
    "dram__bytes_read.sum",
    "gpu__time_duration.sum",
    "sm__warps_active.avg.pct_of_peak_sustained_elapsed",
]


# ── benchmark infrastructure ─────────────────────────────────────────────

@dataclass
class BenchResult:
    name: str
    median_ms: float
    min_ms: float
    max_ms: float
    num_iters: int
    tflops: Optional[float] = None
    bandwidth_gb_s: Optional[float] = None
    extra: dict = field(default_factory=dict)


def cuda_timer(fn, warmup=5, iters=20, sync=True):
    """Time a CUDA function with proper synchronization."""
    for _ in range(warmup):
        fn()
    if sync:
        torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        if sync:
            torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return BenchResult(
        name="", median_ms=times[len(times) // 2],
        min_ms=times[0], max_ms=times[-1], num_iters=iters,
    )


@contextmanager
def nsys_range(name):
    """Mark a range for nsys profiling."""
    if torch.cuda.is_available():
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if torch.cuda.is_available():
            torch.cuda.nvtx.range_pop()


# ── component benchmarks ─────────────────────────────────────────────────

def bench_flashmla_decode(cfg, device):
    """Benchmark FlashMLA dense decode kernel."""
    try:
        from flash_mla import get_mla_metadata, flash_mla_with_kvcache
    except ImportError:
        return BenchResult(name="flashmla_decode", median_ms=-1, min_ms=-1, max_ms=-1,
                           num_iters=0, extra={"skip": "flash_mla not installed"})

    B, H = 32, cfg["num_attention_heads"]
    d_qk = cfg["kv_lora_rank"] + cfg["qk_rope_head_dim"]  # 576 absorbed
    d_v = cfg["kv_lora_rank"]  # 512 absorbed
    seq_len_kv = 4096
    page_size = 64
    num_pages = (B * seq_len_kv + page_size - 1) // page_size

    q = torch.randn(B, 1, H, d_qk, dtype=torch.bfloat16, device=device)
    k_cache = torch.randn(num_pages, page_size, 1, d_qk, dtype=torch.bfloat16, device=device)
    block_table = torch.arange(num_pages, device=device, dtype=torch.int32).view(B, -1)
    cache_seqlens = torch.full((B,), seq_len_kv, dtype=torch.int32, device=device)
    metadata, _ = get_mla_metadata(cache_seqlens, page_size * torch.ones(1, dtype=torch.int32, device=device))

    def run():
        flash_mla_with_kvcache(q, k_cache, block_table, cache_seqlens,
                               head_dim_v=d_v, tile_scheduler_metadata=metadata,
                               softmax_scale=d_qk ** -0.5, causal=False)

    result = cuda_timer(run)
    result.name = "flashmla_decode"
    # TFLOPS: 2 * B * H * 1 * seq_len_kv * d_qk (QK) + 2 * B * H * 1 * seq_len_kv * d_v (PV)
    flops = 2 * B * H * seq_len_kv * (d_qk + d_v)
    result.tflops = flops / (result.median_ms * 1e-3) / 1e12
    result.extra = {"batch": B, "seq_kv": seq_len_kv, "heads": H}
    return result


def bench_deepgemm_mqa_logits(cfg, device):
    """Benchmark DeepGEMM fp8_mqa_logits kernel (DSA indexer)."""
    try:
        import deep_gemm
    except ImportError:
        return BenchResult(name="deepgemm_mqa_logits", median_ms=-1, min_ms=-1, max_ms=-1,
                           num_iters=0, extra={"skip": "deep_gemm not installed"})

    from importlib import import_module
    fp8 = import_module("glm5-kernels-flashmla-deepgemm.fp8_utils")

    seq_len = 1  # decode
    seq_len_kv = 4096
    H = cfg["index_n_heads"]  # 32
    D = cfg["index_head_dim"]  # 128

    # q: raw FP8 tensor [seq_len, H, D] — NOT a tuple (confirmed by debug_all_kernels2.py)
    q = torch.randn(seq_len, H, D, device=device, dtype=torch.bfloat16).to(torch.float8_e4m3fn)
    # kv: tuple of (FP8 tensor [seq_kv, D], 1D scales [seq_kv]) — scales MUST be 1D
    kv = torch.randn(seq_len_kv, D, device=device, dtype=torch.bfloat16)
    kv_fp8 = kv.to(torch.float8_e4m3fn)
    kv_scales = kv.abs().amax(dim=-1).float() / 448.0  # 1D per-row scales [seq_kv]
    weights = torch.randn(seq_len, H, device=device, dtype=torch.float32)
    ks = torch.zeros(seq_len, dtype=torch.int32, device=device)
    ke = torch.full((seq_len,), seq_len_kv, dtype=torch.int32, device=device)

    def run():
        deep_gemm.fp8_mqa_logits(q, (kv_fp8, kv_scales), weights, ks, ke)

    result = cuda_timer(run)
    result.name = "deepgemm_mqa_logits"
    flops = 2 * seq_len * seq_len_kv * H * D
    result.tflops = flops / (result.median_ms * 1e-3) / 1e12
    result.extra = {"seq": seq_len, "seq_kv": seq_len_kv, "heads": H, "dim": D}
    return result


def bench_deepgemm_grouped_gemm(cfg, device):
    """Benchmark DeepGEMM BF16 grouped GEMM (MoE forward).

    Uses BF16 path (confirmed working at 618 TFLOPS / 62.6% MFU on H100).
    FP8 grouped GEMM requires per_block_cast_to_fp8 with TMA-aligned scales
    which has dimension constraints — see benchmark/README.md for details.
    """
    try:
        import deep_gemm
    except ImportError:
        return BenchResult(name="deepgemm_grouped_gemm", median_ms=-1, min_ms=-1, max_ms=-1,
                           num_iters=0, extra={"skip": "deep_gemm not installed"})

    E = cfg["n_routed_experts"]  # 256
    K = cfg["num_experts_per_tok"]  # 8
    D = cfg["hidden_size"]  # 6144
    I = cfg["moe_intermediate_size"]  # 2048
    N = 1024  # tokens per batch
    M = N * K  # total token-expert pairs

    # BF16 tensors — no quantization needed
    a = torch.randn(M, D, device=device, dtype=torch.bfloat16)
    b = torch.randn(E, I, D, device=device, dtype=torch.bfloat16)
    d = torch.empty(M, I, device=device, dtype=torch.bfloat16)

    # Per-row expert index [M] int32 — round-robin assignment
    grouped_layout = (torch.arange(M, device=device, dtype=torch.int32) % E)

    def run():
        deep_gemm.m_grouped_bf16_gemm_nt_contiguous(a, b, d, grouped_layout)

    result = cuda_timer(run, warmup=5, iters=20)
    result.name = "deepgemm_grouped_gemm"
    flops = 2 * M * D * I
    result.tflops = flops / (result.median_ms * 1e-3) / 1e12
    result.extra = {"tokens": N, "experts": E, "topk": K, "D": D, "I": I, "precision": "bf16"}
    return result


def bench_moe_router(cfg, device):
    """Benchmark MoE sigmoid routing (pure PyTorch)."""
    from importlib import import_module
    router = import_module("glm5-kernels-flashmla-deepgemm.moe_router")

    N = 4096
    E = cfg["n_routed_experts"]
    logits = torch.randn(N, E, device=device, dtype=torch.float32)
    bias = torch.randn(E, device=device, dtype=torch.float32)

    def run():
        router.sigmoid_topk_route(logits, bias, top_k=cfg["num_experts_per_tok"],
                                  n_group=cfg["n_group"], topk_group=cfg["topk_group"])

    result = cuda_timer(run, warmup=10, iters=50)
    result.name = "moe_router_sigmoid"
    result.extra = {"tokens": N, "experts": E}
    return result


def bench_single_layer(cfg, device, layer_type="sparse"):
    """Benchmark a single decoder layer (attention + MoE/dense)."""
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashmla-deepgemm.model")
    rope_mod = import_module("glm5-kernels-flashmla-deepgemm.rope_partial")

    layer_idx = 0 if layer_type == "dense" else 3
    cfg_copy = dict(cfg)
    cfg_copy["mlp_layer_types"] = ["dense"] * 3 + ["sparse"] * 75

    layer = kernel_model.DecoderLayer(cfg_copy, layer_idx).to(device).eval()
    # Don't disable kernels — let them use whatever's available

    B, S = 1, 128
    D = cfg["hidden_size"]
    hidden = torch.randn(B, S, D, device=device, dtype=torch.bfloat16)
    rope = rope_mod.RotaryEmbedding(cfg).to(device)
    pos_ids = torch.arange(S, device=device).unsqueeze(0)
    cos, sin = rope(hidden, pos_ids)
    mask = kernel_model.make_causal_mask(S, 0, hidden.dtype, device)

    def run():
        with torch.no_grad():
            layer(hidden, attention_mask=mask, position_embeddings=(cos, sin))

    result = cuda_timer(run, warmup=3, iters=10)
    result.name = f"single_layer_{layer_type}"
    result.extra = {"batch": B, "seq": S, "type": layer_type}
    return result


# ── full model benchmark ─────────────────────────────────────────────────

def bench_full_model_forward(cfg, device, num_layers=4):
    """Benchmark N-layer model forward pass."""
    from importlib import import_module
    kernel_model = import_module("glm5-kernels-flashmla-deepgemm.model")

    cfg_copy = dict(cfg)
    cfg_copy["num_hidden_layers"] = num_layers
    cfg_copy["mlp_layer_types"] = ["dense"] + ["sparse"] * (num_layers - 1)

    model = kernel_model.GlmMoeDsaForCausalLM(cfg_copy).to(device).eval()
    # Keep whatever kernels are available

    B, S = 1, 128
    input_ids = torch.randint(0, cfg["vocab_size"], (B, S), device=device)

    def run():
        with torch.no_grad():
            model(input_ids=input_ids)

    result = cuda_timer(run, warmup=2, iters=5)
    result.name = f"full_model_{num_layers}L"
    result.extra = {"batch": B, "seq": S, "layers": num_layers}
    return result


# ── ncu command generator ────────────────────────────────────────────────

def generate_ncu_commands(script_path: str, output_prefix: str = "glm5_ncu"):
    """Generate ncu command lines for profiling specific kernel types."""
    commands = {}

    # Full profile (all metrics, very slow)
    metrics_str = ",".join(NCU_METRICS.keys())
    commands["full"] = (
        f"ncu --metrics {metrics_str} "
        f"--target-processes all --kernel-name-base function "
        f"-o {output_prefix}_full "
        f"python3 {script_path} --mode ncu"
    )

    # Quick profile (key metrics only)
    quick_str = ",".join(NCU_METRICS_QUICK)
    commands["quick"] = (
        f"ncu --metrics {quick_str} "
        f"--target-processes all --kernel-name-base function "
        f"-o {output_prefix}_quick "
        f"python3 {script_path} --mode ncu"
    )

    # FlashMLA-only (filter by kernel name)
    commands["flashmla"] = (
        f"ncu --metrics {quick_str} "
        f"--kernel-name 'flash_mla|splitkv_mla' "
        f"-o {output_prefix}_flashmla "
        f"python3 {script_path} --mode ncu --component flashmla"
    )

    # DeepGEMM-only
    commands["deepgemm"] = (
        f"ncu --metrics {quick_str} "
        f"--kernel-name 'fp8_mqa_logits|grouped_gemm' "
        f"-o {output_prefix}_deepgemm "
        f"python3 {script_path} --mode ncu --component deepgemm"
    )

    return commands


def generate_nsys_command(script_path: str, output_prefix: str = "glm5_nsys"):
    """Generate nsys command for timeline profiling."""
    return (
        f"nsys profile "
        f"--trace=cuda,nvtx,osrt "
        f"--cuda-memory-usage=true "
        f"--gpuctxsw=true "
        f"--sample=none "
        f"--output={output_prefix} "
        f"--force-overwrite=true "
        f"python3 {script_path} --mode nsys"
    )


# ── multi-GPU utilities ──────────────────────────────────────────────────

def setup_distributed():
    """Initialize torch.distributed for multi-GPU benchmarking."""
    if "RANK" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def bench_nccl_allreduce(hidden_size, device, world_size):
    """Benchmark NCCL all-reduce (simulates tensor parallel reduce)."""
    if world_size <= 1:
        return BenchResult(name="nccl_allreduce", median_ms=0, min_ms=0, max_ms=0,
                           num_iters=0, extra={"skip": "single GPU"})

    B, S = 1, 128
    tensor = torch.randn(B, S, hidden_size, device=device, dtype=torch.bfloat16)

    def run():
        dist.all_reduce(tensor)

    result = cuda_timer(run, warmup=5, iters=20)
    result.name = "nccl_allreduce"
    bytes_transferred = tensor.numel() * tensor.element_size() * 2 * (world_size - 1) / world_size
    result.bandwidth_gb_s = bytes_transferred / (result.median_ms * 1e-3) / 1e9
    result.extra = {"shape": list(tensor.shape), "world_size": world_size}
    return result


# ── main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="GLM-5 H100 Benchmark Harness")
    parser.add_argument("--mode", choices=["bench", "nsys", "ncu", "commands"], default="bench",
                        help="bench=timing, nsys=timeline, ncu=kernel metrics, commands=print profiling commands")
    parser.add_argument("--component", default="all",
                        help="Which component to benchmark: all, flashmla, deepgemm, router, layer, model")
    parser.add_argument("--tp", type=int, default=1, help="Tensor parallel degree")
    parser.add_argument("--full-dims", action="store_true",
                        help="Use full GLM-5 744B dimensions (requires more GPU memory)")
    parser.add_argument("--output", default="bench_results.json", help="Output JSON file")
    args = parser.parse_args()

    if args.mode == "commands":
        script = os.path.abspath(__file__)
        print("=== NCU Profiling Commands ===")
        for name, cmd in generate_ncu_commands(script).items():
            print(f"\n# {name}:")
            print(cmd)
        print(f"\n=== NSYS Timeline Command ===")
        print(generate_nsys_command(script))
        print(f"\n=== NCU Metric Descriptions ===")
        for metric, desc in NCU_METRICS.items():
            print(f"  {metric}:")
            print(f"    {desc}")
        return

    # Setup
    rank, world_size, local_rank = setup_distributed()
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if not torch.cuda.is_available():
        print("ERROR: No CUDA device found. This harness requires H100 GPUs.")
        sys.exit(1)

    props = torch.cuda.get_device_properties(device)
    if rank == 0:
        print(f"GPU: {props.name} (SM{props.major}{props.minor})")
        print(f"Memory: {props.total_memory / 1e9:.1f} GB")
        print(f"World size: {world_size}")
        print(f"Mode: {args.mode}")
        print()

    # Config
    if args.full_dims:
        from .conftest import make_full_cfg
        cfg = make_full_cfg()
    else:
        from .conftest import make_cfg
        cfg = make_cfg(num_layers=4)

    results = []

    # Run benchmarks
    components = {
        "flashmla": lambda: bench_flashmla_decode(cfg, device),
        "deepgemm_indexer": lambda: bench_deepgemm_mqa_logits(cfg, device),
        "deepgemm_moe": lambda: bench_deepgemm_grouped_gemm(cfg, device),
        "router": lambda: bench_moe_router(cfg, device),
        "layer_dense": lambda: bench_single_layer(cfg, device, "dense"),
        "layer_sparse": lambda: bench_single_layer(cfg, device, "sparse"),
        "model": lambda: bench_full_model_forward(cfg, device),
    }

    if world_size > 1:
        components["nccl"] = lambda: bench_nccl_allreduce(cfg["hidden_size"], device, world_size)

    targets = components.keys() if args.component == "all" else [args.component]

    for name in targets:
        if name not in components:
            if rank == 0:
                print(f"Unknown component: {name}")
            continue

        if rank == 0:
            print(f"--- {name} ---")

        with nsys_range(name):
            try:
                result = components[name]()
                results.append(result)
                if rank == 0:
                    line = f"  {result.median_ms:.3f} ms (min={result.min_ms:.3f}, max={result.max_ms:.3f})"
                    if result.tflops and result.tflops > 0:
                        line += f"  {result.tflops:.1f} TFLOPS"
                    if result.bandwidth_gb_s and result.bandwidth_gb_s > 0:
                        line += f"  {result.bandwidth_gb_s:.1f} GB/s"
                    if "skip" in result.extra:
                        line = f"  SKIP: {result.extra['skip']}"
                    print(line)
            except Exception as e:
                if rank == 0:
                    print(f"  ERROR: {e}")
                    import traceback
                    traceback.print_exc()

    # Save results
    if rank == 0 and args.output:
        output = {
            "gpu": props.name,
            "sm_version": f"{props.major}.{props.minor}",
            "gpu_memory_gb": props.total_memory / 1e9,
            "world_size": world_size,
            "config": "full" if args.full_dims else "small",
            "results": [
                {
                    "name": r.name,
                    "median_ms": r.median_ms,
                    "min_ms": r.min_ms,
                    "max_ms": r.max_ms,
                    "tflops": r.tflops,
                    "bandwidth_gb_s": r.bandwidth_gb_s,
                    **r.extra,
                }
                for r in results
            ],
        }
        with open(args.output, "w") as f:
            json.dump(output, f, indent=2)
        print(f"\nResults saved to {args.output}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
