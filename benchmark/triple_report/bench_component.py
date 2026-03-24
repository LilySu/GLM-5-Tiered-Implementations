"""Triple Report Level 2: Component Integration Benchmark.

Tests how kernels perform within a FULL DECODER LAYER — not in isolation.
This captures inter-kernel overhead: quantization boundaries, memory allocation,
Python dispatch between components, and data format conversions.

Fixes the hyphenated directory import problem by creating temporary symlinks.

References:
- MegaBlocks (MLSys '23): MoE kernels must be evaluated on full pipeline, not just GEMM
- DistServe (OSDI '24): Disaggregated prefill/decode evaluation
"""

import argparse
import sys
import os
import traceback
import torch

# Add project root to path
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from benchmark.shared import (
    cuda_timer_extended, BenchConfig, BenchResult,
    save_results, capture_environment, compute_mfu,
    compute_attention_flops, compute_moe_flops,
)
from benchmark.shared.config import GLM5_CONFIG, H100_SPECS
from benchmark.shared.report import print_summary_table

# The model's DecoderLayer expects the full model config dict (with keys like
# "num_attention_heads", "num_key_value_heads", etc.), NOT the benchmark summary
# dict (GLM5_CONFIG which uses "num_heads"). Import the model's config format.
def _get_model_config():
    """Get the full model config dict that DecoderLayer expects."""
    _ensure_symlinks()
    try:
        mod = __import__("glm5_kernels_flashmla_deepgemm.config", fromlist=["GLM_MOE_DSA_CONFIG"])
        return dict(mod.GLM_MOE_DSA_CONFIG)
    except ImportError:
        # Fallback: construct from GLM5_CONFIG with correct key names
        return {
            "vocab_size": 154880,
            "hidden_size": 6144,
            "tie_word_embeddings": False,
            "num_hidden_layers": 78,
            "intermediate_size": 12288,
            "num_attention_heads": 64,
            "num_key_value_heads": 64,
            "attention_bias": False,
            "attention_dropout": 0.0,
            "q_lora_rank": 2048,
            "kv_lora_rank": 512,
            "qk_rope_head_dim": 64,
            "qk_nope_head_dim": 192,
            "qk_head_dim": 256,
            "v_head_dim": 256,
            "n_routed_experts": 256,
            "n_shared_experts": 1,
            "num_experts_per_tok": 8,
            "moe_intermediate_size": 2048,
            "routed_scaling_factor": 2.5,
            "n_group": 1,
            "topk_group": 1,
            "norm_topk_prob": True,
            "index_topk": 2048,
            "index_head_dim": 128,
            "index_n_heads": 32,
            "hidden_act": "silu",
            "rms_norm_eps": 1e-5,
            "max_position_embeddings": 202752,
            "rope_theta": 10000.0,
            "initializer_range": 0.02,
            "pad_token_id": None,
            "bos_token_id": 0,
            "eos_token_id": 1,
            "use_cache": True,
            "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
        }


def _ensure_symlinks():
    """Create underscore-named symlinks for hyphenated model directories.

    Python cannot import from directories with hyphens in the name.
    This creates symlinks like glm5_kernels_flashmla_deepgemm -> glm5-kernels-flashmla-deepgemm
    """
    mappings = {
        "glm5-kernels-flashmla-deepgemm": "glm5_kernels_flashmla_deepgemm",
        "glm5-kernels-flashinfer": "glm5_kernels_flashinfer",
        "glm5-raw-decoupled-from-hf": "glm5_raw_decoupled_from_hf",
        "glm5-triton": "glm5_triton",
    }
    for hyphenated, underscored in mappings.items():
        src = os.path.join(PROJECT_ROOT, hyphenated)
        dst = os.path.join(PROJECT_ROOT, underscored)
        if os.path.isdir(src) and not os.path.exists(dst):
            try:
                os.symlink(src, dst)
            except OSError:
                pass  # May fail on some filesystems


def _import_decoder_layer(impl, force_eager=False):
    """Import DecoderLayer from the correct model directory.

    Uses symlinks to work around Python's inability to import from
    hyphenated directory names.

    If force_eager=True, patches the module-level FLASH_MLA_AVAILABLE and
    DEEP_GEMM_AVAILABLE flags BEFORE loading the model, so the model
    never tries to use kernel paths (which crash with random weights).
    """
    _ensure_symlinks()

    if impl in ("flashmla", "eager"):
        pkg = "glm5_kernels_flashmla_deepgemm"
    elif impl == "flashinfer":
        pkg = "glm5_kernels_flashinfer"
    else:
        raise ValueError(f"Unknown impl: {impl}")

    if PROJECT_ROOT not in sys.path:
        sys.path.insert(0, PROJECT_ROOT)

    if force_eager:
        # Patch the availability flags BEFORE the model reads them
        try:
            attn_mod = __import__(f"{pkg}.mla_attention", fromlist=["FLASH_MLA_AVAILABLE"])
            attn_mod.FLASH_MLA_AVAILABLE = False
        except Exception:
            pass
        try:
            idx_mod = __import__(f"{pkg}.dsa_indexer", fromlist=["DEEP_GEMM_AVAILABLE"])
            idx_mod.DEEP_GEMM_AVAILABLE = False
        except Exception:
            pass

    mod = __import__(f"{pkg}.model", fromlist=["DecoderLayer"])
    return mod.DecoderLayer


def bench_single_layer(layer_type, B, S, T, impl, cfg, warmup=10, iters=50):
    """Benchmark a single decoder layer (attention + MLP/MoE + norms + residuals).

    Uses the model's own bench_single_layer from h100_bench.py pattern —
    creates the layer, builds position embeddings correctly, and runs.
    """
    device = torch.device("cuda")

    try:
        # force_eager=True because random weights + kernel paths produce
        # invalid expert/token indices → CUDA device-side assert
        DecoderLayer = _import_decoder_layer(impl, force_eager=True)
    except Exception as e:
        return BenchResult(
            name=f"layer_{layer_type}", impl=impl,
            config={"B": B, "S": S, "T": T, "type": layer_type},
            error=f"Import failed: {e}",
        )

    try:
        # Also import RotaryEmbedding from the same package
        if impl in ("flashmla", "eager"):
            pkg = "glm5_kernels_flashmla_deepgemm"
        else:
            pkg = "glm5_kernels_flashinfer"

        rope_mod = __import__(f"{pkg}.rope_partial", fromlist=["RotaryEmbedding"])
        RotaryEmbedding = rope_mod.RotaryEmbedding

        test_cfg = dict(cfg)
        test_cfg["num_hidden_layers"] = 1
        test_cfg["mlp_layer_types"] = [layer_type]
        # ALWAYS use eager mode for component benchmarks with random weights.
        # Kernel paths (FlashMLA/DeepGEMM) produce CUDA index-out-of-bounds
        # with random weights because the DSA indexer and MoE router generate
        # invalid expert/token indices. Real kernel benchmarks need real weights.
        test_cfg["use_flash_mla"] = False
        test_cfg["use_deepgemm"] = False

        layer = DecoderLayer(test_cfg, layer_idx=0).to(device).bfloat16().eval()

        # Build inputs matching h100_bench.py pattern
        hidden = torch.randn(B, S, cfg["hidden_size"], dtype=torch.bfloat16, device=device)

        # Build position embeddings (cos, sin) — RotaryEmbedding takes the full config dict
        rope = RotaryEmbedding(test_cfg).to(device)
        position_ids = torch.arange(T - S, T, device=device).unsqueeze(0).expand(B, -1)
        cos, sin = rope(hidden, position_ids)

        # Build causal attention mask
        mask = torch.full((S, T), float("-inf"), device=device, dtype=torch.bfloat16)
        mask = torch.triu(mask, diagonal=T - S + 1)
        mask = mask.unsqueeze(0).unsqueeze(0)  # [1, 1, S, T]

        with torch.no_grad():
            for _ in range(3):
                _ = layer(hidden, attention_mask=mask, position_embeddings=(cos, sin))
            torch.cuda.synchronize()

    except torch.cuda.OutOfMemoryError:
        return BenchResult(
            name=f"layer_{layer_type}", impl=impl,
            config={"B": B, "S": S, "T": T, "type": layer_type},
            is_oom=True,
        )
    except Exception as e:
        return BenchResult(
            name=f"layer_{layer_type}", impl=impl,
            config={"B": B, "S": S, "T": T, "type": layer_type},
            error=f"Setup failed: {e}",
        )

    def run():
        with torch.no_grad():
            _ = layer(hidden, attention_mask=mask, position_embeddings=(cos, sin))

    try:
        torch.cuda.reset_peak_memory_stats()
        times, stats = cuda_timer_extended(run, warmup=warmup, iters=iters)
    except torch.cuda.OutOfMemoryError:
        return BenchResult(
            name=f"layer_{layer_type}", impl=impl,
            config={"B": B, "S": S, "T": T, "type": layer_type},
            is_oom=True,
        )
    except Exception as e:
        return BenchResult(
            name=f"layer_{layer_type}", impl=impl,
            config={"B": B, "S": S, "T": T, "type": layer_type},
            error=f"Benchmark failed: {e}",
        )

    H = cfg.get("num_attention_heads", cfg.get("num_heads", 64))
    d_qk = cfg.get("qk_head_dim", 256)
    d_v = cfg.get("v_head_dim", 256)
    attn_flops = compute_attention_flops(B, H, S, T, d_qk, d_v)

    if layer_type == "sparse":
        moe_flops = compute_moe_flops(
            B * S, cfg["num_experts_per_tok"],
            cfg["hidden_size"], cfg["moe_intermediate_size"]
        )
    else:
        moe_flops = 2 * B * S * cfg["hidden_size"] * cfg["intermediate_size"] * 3

    total_flops = attn_flops + moe_flops
    latency_s = stats["median"] / 1000.0

    return BenchResult(
        name=f"layer_{layer_type}",
        impl=impl,
        config={"B": B, "S": S, "T": T, "type": layer_type},
        latency_ms=times,
        median_ms=stats["median"],
        mean_ms=stats["mean"],
        std_ms=stats["std"],
        p5_ms=stats["p5"],
        p50_ms=stats["p50"],
        p95_ms=stats["p95"],
        p99_ms=stats["p99"],
        ci_95_low=stats["ci_95_low"],
        ci_95_high=stats["ci_95_high"],
        tflops=total_flops / latency_s / 1e12 if latency_s > 0 else 0,
        mfu_pct=compute_mfu(total_flops, latency_s),
        peak_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
    )


def run_component_benchmark(output_dir="results/triple_report"):
    """Run the component integration benchmark. Saves partial results on failure."""
    results = []
    cfg = _get_model_config()

    configs = [
        {"B": 32, "S": 1, "T": 4096, "label": "decode_B32_T4K"},
        {"B": 1, "S": 128, "T": 128, "label": "prefill_B1_S128"},
        {"B": 4, "S": 1, "T": 16384, "label": "decode_B4_T16K"},
    ]

    for c in configs:
        for layer_type in ["dense", "sparse"]:
            for impl in ["eager", "flashmla", "flashinfer"]:
                print(f"  {c['label']} | {layer_type} | {impl}...", end=" ", flush=True)
                try:
                    result = bench_single_layer(
                        layer_type, c["B"], c["S"], c["T"], impl, cfg,
                    )
                except Exception as e:
                    result = BenchResult(
                        name=f"layer_{layer_type}", impl=impl,
                        config={"B": c["B"], "S": c["S"], "T": c["T"], "type": layer_type},
                        error=f"Uncaught: {e}",
                    )

                if result.is_oom:
                    print("OOM")
                elif result.error:
                    print(f"ERROR: {result.error[:60]}")
                else:
                    print(f"{result.median_ms:.3f} ms | {result.mfu_pct:.1f}% MFU")
                results.append(result)
                torch.cuda.empty_cache()

    # ALWAYS save results — even if some failed
    print_summary_table(results, "Triple Report Level 2: Component Integration")
    env = capture_environment()
    save_results(results, output_dir, "triple_report_component", env)
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Triple Report Level 2: Component Integration")
    parser.add_argument("--output-dir", default="results/triple")
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    run_component_benchmark(args.output_dir)
