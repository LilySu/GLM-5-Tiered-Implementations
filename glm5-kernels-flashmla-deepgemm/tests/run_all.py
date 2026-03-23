"""Run all tests for glm5-kernels-flashmla-deepgemm.

Usage:
    # CPU tests only (no GPU needed):
    python3 -m glm5-kernels-flashmla-deepgemm.tests.run_all

    # Include H100 kernel tests (requires SM90 + flash-mla + deep-gemm):
    python3 -m glm5-kernels-flashmla-deepgemm.tests.run_all --h100

    # H100 3-way benchmark:
    python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench_3way

    # H100 kernel-level profiling:
    python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode commands
"""

import sys
import os
import importlib
import traceback


# CPU tests — run anywhere, no CUDA required
CPU_TEST_MODULES = [
    # Original 6 equivalence tests
    ("test_equivalence", [
        "test_moe_router", "test_fp8_utils", "test_dsa_indexer",
        "test_rmsnorm", "test_mla_attention", "test_full_model",
    ]),
    # Test 7: cu_seqlens
    ("test_deepgemm_cu_seqlens", [
        "test_cu_seqlens_prefill", "test_cu_seqlens_decode", "test_cu_seqlens_mismatched",
    ]),
    # Test 8: KV cache
    ("test_kv_cache", [
        "test_kvcache_multistep", "test_kvcache_reset", "test_paged_kvcache_allocation",
    ]),
    # Test 9: DSA mask
    ("test_dsa_mask", [
        "test_dsa_mask_basic", "test_dsa_mask_with_causal",
    ]),
    # Test 10: MoE dispatch
    ("test_moe_expert_dispatch", [
        "test_expert_dispatch_single_expert", "test_expert_dispatch_no_tokens",
    ]),
    # Test 11: FP8 layout
    ("test_fp8_layout", [
        "test_flashmla_kv_byte_layout", "test_fp8_nope_roundtrip",
    ]),
    # Test 12: Autoregressive decode
    ("test_autoregressive_decode", [
        "test_autoregressive_3step",
    ]),
    # Test 13: Group routing
    ("test_group_routing", [
        "test_group_routing_filters_groups", "test_group_routing_vs_flat",
    ]),
    # Test 14: Gradient flow
    ("test_gradient_flow", [
        "test_gradient_flow",
    ]),
    # Test 15: State dict compat
    ("test_state_dict_compat", [
        "test_state_dict_keys_match", "test_state_dict_cross_load", "test_state_dict_shapes_match",
    ]),
    # Test 16: Edge cases
    ("test_edge_cases", [
        "test_single_token_forward", "test_topk_exceeds_seq_len",
        "test_moe_with_shared_expert", "test_empty_expert_assignment",
    ]),
]


# H100 tests — require SM90 GPU + flash-mla + deep-gemm installed
H100_TEST_MODULES = [
    # Kernel correctness (require flash-mla / deep-gemm)
    ("h100_test_flashmla_kernels", [
        "h100_test_flashmla_dense_decode",
        "h100_test_flashmla_sparse_prefill",
        "h100_test_flashmla_fp8_kv_decode",
    ]),
    ("h100_test_deepgemm_kernels", [
        "h100_test_deepgemm_fp8_mqa_logits",
        "h100_test_deepgemm_fp8_mqa_logits_glm5_dims",
        "h100_test_deepgemm_grouped_gemm_contiguous",
        "h100_test_deepgemm_grouped_gemm_masked",
    ]),
    # ── Non-graph tests first (these must not be poisoned by graph failures) ──
    # Cat 2: TMA Verification
    ("h100_test_tma", [
        "h100_test_tma_bandwidth_flashmla",
        "h100_test_tma_bandwidth_deepgemm",
    ]),
    # Cat 3: Memory Allocation & Peak
    ("h100_test_memory", [
        "h100_test_memory_peak_single_layer",
        "h100_test_memory_kv_cache_scaling",
        "h100_test_memory_no_leak_decode",
    ]),
    # Cat 4: FP8 Numeric Edge Cases
    ("h100_test_fp8_edge_cases", [
        "h100_test_fp8_overflow_detection",
        "h100_test_fp8_zero_handling",
        "h100_test_fp8_subnormal_precision",
        "h100_test_fp8_flashmla_kv_scale_correctness",
    ]),
    # Cat 7: Deterministic Execution
    ("h100_test_determinism", [
        "h100_test_deterministic_topk",
        "h100_test_deterministic_full_decode",
        "h100_test_deterministic_dsa_indexer",
    ]),
    # Cat 8: Sparse Attention Patterns
    ("h100_test_sparse_patterns", [
        "h100_test_sparse_causality",
        "h100_test_sparse_recency_bias",
        "h100_test_sparse_non_degeneracy",
        "h100_test_sparse_jaccard_stability",
    ]),
    # Cat 9: Precision Boundary Stress
    ("h100_test_precision_chain", [
        "h100_test_precision_chain_roundtrips",
        "h100_test_precision_full_pipeline",
    ]),
    # Cat 10: Thermal Throttling Detection
    ("h100_test_thermal", [
        "h100_test_thermal_sustained_gemm",
        "h100_test_thermal_clock_frequency",
    ]),
    # ── CUDA Graph tests LAST (can poison CUDA context on failure) ──
    # Cat 6: Kernel Launch Overhead (includes graph tests)
    ("h100_test_launch_overhead", [
        "h100_test_launch_overhead_empty_kernels",
        "h100_test_launch_overhead_per_layer",
        "h100_test_launch_overhead_graph_vs_eager_model",
    ]),
    # Cat 1: CUDA Graph Capture & Replay
    ("h100_test_cuda_graph", [
        "h100_test_cuda_graph_capture_model",
        "h100_test_cuda_graph_sparse_index_update",
        "h100_test_cuda_graph_speedup",
    ]),
]

# Cat 5: Multi-GPU (requires torchrun, run separately)
# torchrun --nproc_per_node=2 -m glm5-kernels-flashmla-deepgemm.tests.h100_test_multi_gpu


def _run_test_in_subprocess(module_name, fn_name, package):
    """Run a single test in an isolated subprocess to prevent CUDA context poisoning.

    Failed CUDA graph captures corrupt the entire process's CUDA context.
    gc.collect()/empty_cache()/synchronize() CANNOT fix this in PyTorch 2.8.0+.
    The ONLY reliable fix is process isolation.
    """
    import subprocess
    cmd = [
        sys.executable, "-c",
        f"import sys; sys.path.insert(0, '.'); "
        f"from importlib import import_module; "
        f"mod = import_module('.{module_name}', package='{package}'); "
        f"fn = getattr(mod, '{fn_name}'); "
        f"result = fn(); "
        f"sys.exit(0 if result else 1)"
    ]
    try:
        proc = subprocess.run(
            cmd, capture_output=True, text=True, timeout=300,
            cwd=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")),
        )
        # Print the subprocess output
        if proc.stdout:
            for line in proc.stdout.strip().split('\n'):
                print(line)
        if proc.stderr:
            # Only print stderr if it contains real errors, not warnings
            for line in proc.stderr.strip().split('\n'):
                if 'Error' in line or 'FAIL' in line or 'Traceback' in line:
                    print(line)
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        print(f"  ERROR {module_name}.{fn_name}: timed out after 300s")
        return False
    except Exception as e:
        print(f"  ERROR {module_name}.{fn_name}: subprocess failed: {e}")
        return False


# Tests that use CUDA graphs and can poison the CUDA context if they fail
GRAPH_TESTS = {
    "h100_test_cuda_graph",
    "h100_test_launch_overhead",
}


def run_test_list(test_modules, label):
    all_results = {}
    total = 0
    passed = 0
    failed = 0
    errors = 0

    for module_name, test_fns in test_modules:
        use_subprocess = module_name in GRAPH_TESTS
        if not use_subprocess:
            mod = importlib.import_module(f".{module_name}", package="glm5-kernels-flashmla-deepgemm.tests")

        for fn_name in test_fns:
            total += 1
            full_name = f"{module_name}.{fn_name}"
            try:
                if use_subprocess:
                    # Run in subprocess to isolate CUDA graph failures
                    result = _run_test_in_subprocess(
                        module_name, fn_name, "glm5-kernels-flashmla-deepgemm.tests")
                else:
                    fn = getattr(mod, fn_name)
                    result = fn()
                all_results[full_name] = result
                if result:
                    passed += 1
                else:
                    failed += 1
            except Exception as e:
                print(f"  ERROR {full_name}: {e}")
                traceback.print_exc()
                all_results[full_name] = False
                errors += 1

    print(f"\n{'='*70}")
    print(f"{label}: {passed}/{total} passed, {failed} failed, {errors} errors")
    print("=" * 70)
    for name, result in all_results.items():
        status = "PASS" if result else "FAIL"
        print(f"  {status}  {name}")

    return failed == 0 and errors == 0


def main():
    include_h100 = "--h100" in sys.argv

    print("=" * 70)
    print("GLM-5 Kernel Test Suite")
    print("=" * 70)

    ok = run_test_list(CPU_TEST_MODULES, "CPU TESTS (no GPU required)")

    if include_h100:
        import torch
        if not torch.cuda.is_available():
            print("\nWARN: --h100 requested but no CUDA device found. Skipping H100 tests.")
        else:
            props = torch.cuda.get_device_properties(0)
            print(f"\nGPU: {props.name} (SM{props.major}{props.minor})")
            if props.major != 9:
                print("WARN: Not an SM90 GPU. H100 tests may skip or fail.")
            h100_ok = run_test_list(H100_TEST_MODULES, "H100 KERNEL TESTS (SM90 required)")
            ok = ok and h100_ok
    else:
        print("\nNote: Run with --h100 to include FlashMLA/DeepGEMM kernel tests on H100.")

    print()
    if ok:
        print("All tests passed.")
    else:
        print("Some tests FAILED.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
