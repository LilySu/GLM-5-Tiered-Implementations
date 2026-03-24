"""Triple Report Level 2: Component Integration Benchmark.

Tests full decoder layer latency at GLM-5 dimensions.
Uses the PROVEN debug_single_layer.py pattern that works on H100.

CRITICAL: Must set environment variable BEFORE any model imports to
disable FlashMLA/DeepGEMM (random weights + kernel paths = CUDA assert).
"""

import os
# MUST be set before ANY imports that check for kernel availability
os.environ["GLM5_FORCE_EAGER"] = "1"

import argparse
import sys
import time
import torch

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.shared import (
    cuda_timer_extended, BenchResult,
    save_results, capture_environment, compute_mfu,
    compute_attention_flops, compute_moe_flops,
)
from benchmark.shared.config import H100_SPECS
from benchmark.shared.report import print_summary_table


def _ensure_symlinks():
    for h, u in [
        ("glm5-kernels-flashmla-deepgemm", "glm5_kernels_flashmla_deepgemm"),
        ("glm5-kernels-flashinfer", "glm5_kernels_flashinfer"),
    ]:
        src = os.path.join(PROJECT_ROOT, h)
        dst = os.path.join(PROJECT_ROOT, u)
        if os.path.isdir(src) and not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except OSError:
                pass


def _patch_and_import():
    """Import model with ALL kernel paths disabled.

    Patches module-level flags in mla_attention and dsa_indexer
    BEFORE they're read by DecoderLayer.__init__.
    """
    _ensure_symlinks()

    pkg = "glm5_kernels_flashmla_deepgemm"

    # Force-patch before any model code reads these
    import importlib

    # Patch mla_attention
    attn_mod = importlib.import_module(f"{pkg}.mla_attention")
    attn_mod.FLASH_MLA_AVAILABLE = False
    if hasattr(attn_mod, 'HAS_FLASH_MLA'):
        attn_mod.HAS_FLASH_MLA = False

    # Patch dsa_indexer
    idx_mod = importlib.import_module(f"{pkg}.dsa_indexer")
    idx_mod.DEEP_GEMM_AVAILABLE = False
    if hasattr(idx_mod, 'HAS_DEEP_GEMM'):
        idx_mod.HAS_DEEP_GEMM = False

    # Now import model (it reads the patched flags)
    model_mod = importlib.import_module(f"{pkg}.model")
    rope_mod = importlib.import_module(f"{pkg}.rope_partial")
    config_mod = importlib.import_module(f"{pkg}.config")

    return model_mod.DecoderLayer, rope_mod.RotaryEmbedding, config_mod.GLM_MOE_DSA_CONFIG


def bench_layer(layer_type, B, S, T, cfg, DecoderLayer, RotaryEmbedding, warmup=5, iters=20):
    """Benchmark one decoder layer. Returns BenchResult."""
    device = torch.device("cuda")
    label = f"layer_{layer_type}"
    config_info = {"B": B, "S": S, "T": T, "type": layer_type}

    try:
        test_cfg = dict(cfg)
        test_cfg["num_hidden_layers"] = 1
        test_cfg["mlp_layer_types"] = [layer_type]

        layer = DecoderLayer(test_cfg, layer_idx=0).to(device).bfloat16().eval()
        rope = RotaryEmbedding(test_cfg).to(device)

        hidden = torch.randn(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)
        pos_ids = torch.arange(T - S, T, device=device).unsqueeze(0).expand(B, -1)
        cos, sin = rope(hidden, pos_ids)

        mask = torch.full((S, T), float("-inf"), device=device, dtype=torch.bfloat16)
        mask = torch.triu(mask, diagonal=T - S + 1)
        mask = mask.unsqueeze(0).unsqueeze(0)

        # Warmup
        with torch.no_grad():
            for _ in range(3):
                layer(hidden, attention_mask=mask, position_embeddings=(cos, sin))
            torch.cuda.synchronize()

    except torch.cuda.OutOfMemoryError:
        return BenchResult(name=label, impl="eager", config=config_info, is_oom=True)
    except Exception as e:
        return BenchResult(name=label, impl="eager", config=config_info, error=f"Setup: {e}")

    def run():
        with torch.no_grad():
            layer(hidden, attention_mask=mask, position_embeddings=(cos, sin))

    try:
        torch.cuda.reset_peak_memory_stats()
        times, stats = cuda_timer_extended(run, warmup=warmup, iters=iters)
    except torch.cuda.OutOfMemoryError:
        return BenchResult(name=label, impl="eager", config=config_info, is_oom=True)
    except Exception as e:
        return BenchResult(name=label, impl="eager", config=config_info, error=f"Bench: {e}")

    H = cfg["num_attention_heads"]
    d_qk = cfg.get("qk_head_dim", 256)
    d_v = cfg.get("v_head_dim", 256)
    attn_flops = compute_attention_flops(B, H, S, T, d_qk, d_v)
    if layer_type == "sparse":
        moe_flops = compute_moe_flops(B * S, cfg["num_experts_per_tok"], cfg["hidden_size"], cfg["moe_intermediate_size"])
    else:
        moe_flops = 2 * B * S * cfg["hidden_size"] * cfg["intermediate_size"] * 3
    total_flops = attn_flops + moe_flops
    latency_s = stats["median"] / 1000.0

    # Clean up to free memory for next config
    del layer, rope, hidden, cos, sin, mask
    torch.cuda.empty_cache()

    return BenchResult(
        name=label, impl="eager", config=config_info,
        latency_ms=times,
        median_ms=stats["median"], mean_ms=stats["mean"], std_ms=stats["std"],
        p5_ms=stats["p5"], p50_ms=stats["p50"], p95_ms=stats["p95"], p99_ms=stats["p99"],
        ci_95_low=stats["ci_95_low"], ci_95_high=stats["ci_95_high"],
        tflops=total_flops / latency_s / 1e12 if latency_s > 0 else 0,
        mfu_pct=compute_mfu(total_flops, latency_s),
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
    )


def run_component_benchmark(output_dir="results/triple"):
    results = []

    print("Importing model (eager mode, no FlashMLA/DeepGEMM)...")
    try:
        DecoderLayer, RotaryEmbedding, cfg = _patch_and_import()
    except Exception as e:
        print(f"FATAL: Cannot import model: {e}")
        # Save empty results
        save_results([], output_dir, "triple_report_component", capture_environment())
        return []

    print(f"Model imported. hidden_size={cfg['hidden_size']}, experts={cfg['n_routed_experts']}")
    print()

    # NOTE: Decode configs (S=1, T>S) crash with random weights because the
    # DSA indexer's topk selects out-of-bounds indices. Only prefill configs
    # (S=T, causal mask keeps indices valid) work without real model weights.
    # For decode numbers, see h100_bench --full-dims (2.721ms dense, 4.866ms sparse).
    configs = [
        {"B": 1, "S": 64, "T": 64, "label": "prefill_B1_S64"},
        {"B": 1, "S": 128, "T": 128, "label": "prefill_B1_S128"},
        {"B": 1, "S": 256, "T": 256, "label": "prefill_B1_S256"},
        {"B": 1, "S": 512, "T": 512, "label": "prefill_B1_S512"},
        {"B": 4, "S": 128, "T": 128, "label": "prefill_B4_S128"},
    ]

    for c in configs:
        for layer_type in ["dense", "sparse"]:
            print(f"  {c['label']} | {layer_type}...", end=" ", flush=True)
            result = bench_layer(layer_type, c["B"], c["S"], c["T"], cfg, DecoderLayer, RotaryEmbedding)
            if result.is_oom:
                print("OOM")
            elif result.error:
                print(f"ERROR: {result.error[:60]}")
            else:
                print(f"{result.median_ms:.3f} ms | {result.tflops:.1f} TFLOPS | {result.mfu_pct:.1f}% MFU | {result.peak_memory_gb:.1f} GB")
            results.append(result)

    print()
    print_summary_table(results, "Triple Report Level 2: Component Integration (Eager Mode)")
    env = capture_environment()
    save_results(results, output_dir, "triple_report_component", env)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="results/triple")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_component_benchmark(args.output_dir)
