# Benchmark CLI Commands List

All commands assume you have already run `source /workspace/GLM-5-Decoupled-From-HuggingFace/setup.sh` to set up kernel packages.

## Never run (high priority)

MoE quick sweep with DeepGEMM + patched fp8_mqa_logits (~3 min):
```
cd /workspace/GLM-5-Decoupled-From-HuggingFace && python3 -m benchmark.moe_sweep.bench_moe --quick --output-dir results/moe_with_dg/
```

FlashMLA vs FlashInfer head-to-head (~15 min):
```
cd /workspace/GLM-5-Decoupled-From-HuggingFace && python3 benchmark_head_to_head.py --experiment component --output-dir results/h2h/
```

3-way comparison — PyTorch vs Triton vs Kernels (~15 min):
```
cd /workspace/GLM-5-Decoupled-From-HuggingFace && python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench_3way --full-dims
```

## Previously run without DeepGEMM (worth re-running)

MFU ceiling (~5 min) — previous run on this pod had dg=not installed:
```
cd /workspace/GLM-5-Decoupled-From-HuggingFace && python3 -m benchmark.mfu_ceiling.bench_mfu --output-dir results/mfu/
```

FP8 Pareto (~5 min) — previous run on this pod had no DeepGEMM:
```
cd /workspace/GLM-5-Decoupled-From-HuggingFace && python3 -m benchmark.fp8_pareto.bench_fp8 --output-dir results/fp8/
```

## Lower priority (long)

Full MoE sweep, 1440 configs (~2 hours) — only worth it for SC'25-style parametric data:
```
cd /workspace/GLM-5-Decoupled-From-HuggingFace && python3 -m benchmark.moe_sweep.bench_moe --output-dir results/moe_full/
```

## Serving profile experiments (NEW)

Prefill vs Decode phase separation (~20 min):
```
cd /workspace/GLM-5-Decoupled-From-HuggingFace && python3 -m benchmark.bench_serving_profile --experiment prefill-decode --output-dir results/serving_profile/
```

MoE expert routing skew — uniform vs Zipf (~10 min):
```
cd /workspace/GLM-5-Decoupled-From-HuggingFace && python3 -m benchmark.bench_serving_profile --experiment routing-skew --output-dir results/serving_profile/
```

KV cache memory footprint and MLA compression ratio (~2 min):
```
cd /workspace/GLM-5-Decoupled-From-HuggingFace && python3 -m benchmark.bench_serving_profile --experiment memory --output-dir results/serving_profile/
```

All three serving profile experiments (~30 min):
```
cd /workspace/GLM-5-Decoupled-From-HuggingFace && python3 -m benchmark.bench_serving_profile --experiment all --output-dir results/serving_profile/
```

## DSA indexer context scaling (NEW)

DSA indexer latency scaling at GLM-5 Table 6 context lengths (~10 min):
```
cd /workspace/GLM-5-Decoupled-From-HuggingFace && python3 -m benchmark.bench_dsa_context_scaling --experiment all --output-dir results/dsa_scaling/
```

Indexer-only scaling curve (~5 min):
```
cd /workspace/GLM-5-Decoupled-From-HuggingFace && python3 -m benchmark.bench_dsa_context_scaling --experiment indexer-scaling --output-dir results/dsa_scaling/
```

Indexer as fraction of total decode latency (~5 min):
```
cd /workspace/GLM-5-Decoupled-From-HuggingFace && python3 -m benchmark.bench_dsa_context_scaling --experiment indexer-fraction --output-dir results/dsa_scaling/
```

## Already completed

- `h100_bench --full-dims` — FlashMLA 228 TFLOPS, DeepGEMM MoE 606 TFLOPS, dense layer 2.7ms, sparse layer 4.8ms
- `bench_micro` — 16 component microbenchmarks
- `bench_component` — full decoder layer (multiple runs)
- `bench_e2e` — 4 serving scenarios (chatbot, code_assist, long_doc_qa, agentic_swe)
- `bench_mfu` — MFU ceiling (with DeepGEMM on previous pod)
- `bench_fp8` — FP8 Pareto (with DeepGEMM on previous pod)
- `bench_moe --quick` — MoE quick sweep (with DeepGEMM on previous pod)
- `fix_kernels_h100` — all 3 kernel API checks passed
