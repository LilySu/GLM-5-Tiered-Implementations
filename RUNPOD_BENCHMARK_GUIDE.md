# RunPod H100 Benchmark Guide

Step-by-step commands for benchmarking GLM-5 kernel implementations on a RunPod H100 instance.

---

## Prerequisites

- **GPU:** 1× NVIDIA H100 80GB SXM5 (SM90)
- **CUDA:** 12.8+ (required by FlashMLA and DeepGEMM)
- **OS:** Ubuntu 22.04+ (RunPod default)
- **Python:** 3.10+

---

## Step 1: SSH into RunPod and Verify GPU

```bash
# After SSH'ing into your RunPod instance:

# Verify H100 is present and CUDA version
nvidia-smi
# Expected: "NVIDIA H100 80GB HBM3" and "CUDA Version: 12.x"

# Check SM version (must be 9.0 for H100)
nvidia-smi --query-gpu=compute_cap --format=csv,noheader
# Expected: 9.0

# Check available memory
nvidia-smi --query-gpu=memory.total,memory.free --format=csv
# Expected: 81559 MiB total, ~80000 MiB free
```

## Step 2: Clone the Repo

```bash
cd /workspace  # RunPod's persistent storage
git clone https://github.com/YOUR_REPO/glm5.git
cd glm5
```

## Step 3: Install PyTorch + CUDA 12.8

```bash
# If PyTorch isn't pre-installed (check with: python3 -c "import torch; print(torch.version.cuda)")
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu128

# Verify
python3 -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'SM: {torch.cuda.get_device_capability(0)}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB')
"
```

## Step 4: Install Kernel Libraries

### FlashMLA (from source — no pip wheel)
```bash
cd /workspace
git clone --recurse-submodules https://github.com/deepseek-ai/FlashMLA.git
cd FlashMLA
pip install -v .
cd /workspace/glm5

# Verify
python3 -c "import flash_mla; print('FlashMLA installed')"
```

### DeepGEMM (JIT-compiled — no build step)
```bash
cd /workspace
git clone https://github.com/deepseek-ai/DeepGEMM.git
cd DeepGEMM
pip install -e .
cd /workspace/glm5

# Verify (first import triggers JIT, may take 10-60s)
python3 -c "import deep_gemm; print('DeepGEMM installed')"
```

### FlashInfer
```bash
# NOTE: The package name is "flashinfer-python" (NOT "flashinfer")

# Option A: pip install (stable release)
pip install flashinfer-python flashinfer-cubin
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128

# Option B: pip install (nightly — for latest torch/CUDA combos like torch 2.8)
pip install -U --pre flashinfer-python --index-url https://flashinfer.ai/whl/nightly/ --no-deps
pip install flashinfer-python flashinfer-cubin
pip install flashinfer-jit-cache --index-url https://flashinfer.ai/whl/cu128

# Option C: from source (guaranteed to work with any torch/CUDA)
cd /workspace
git clone https://github.com/flashinfer-ai/flashinfer.git
cd flashinfer && pip install -e . -v  # Takes 5-10 min to compile
cd /workspace/glm5

# Verify
python3 -c "import flashinfer; print('FlashInfer installed')"
```

### Triton (usually pre-installed with PyTorch)
```bash
python3 -c "import triton; print(f'Triton: {triton.__version__}')"
# If missing: pip install triton
```

## Step 5: Install Python Dependencies

```bash
cd /workspace/glm5
pip install numpy  # for benchmark statistics (bootstrap CI)
```

## Step 6: Lock GPU Clocks (CRITICAL for reproducible benchmarks)

```bash
# Lock SM and memory clocks to maximum to prevent thermal throttling variance
sudo nvidia-smi -pm 1                           # Enable persistence mode
sudo nvidia-smi -lgc 2100,2100                  # Lock graphics clock to max
sudo nvidia-smi --power-limit 700               # Set power limit (H100 SXM default TDP)

# Verify clocks are locked
nvidia-smi -q -d CLOCK | grep -A2 "Max Clocks"
nvidia-smi -q -d CLOCK | grep -A2 "Applications Clocks"

# Check temperature baseline (should be <45°C idle)
nvidia-smi --query-gpu=temperature.gpu --format=csv,noheader
```

> **Why lock clocks?** Without locking, H100 dynamically adjusts frequency based on temperature and power. This causes 5-15% variance between runs. Academic benchmarks require locked clocks (see FlashAttention-3 methodology).

## Step 7: Run CPU Tests First (Verify Setup)

```bash
cd /workspace/glm5

# Run all 29 CPU tests (~10 seconds, no GPU needed)
python3 -m glm5-kernels-flashmla-deepgemm.tests.run_all
# Expected: "29/29 tests passed"

# If flashinfer variant exists:
python3 -m glm5-kernels-flashinfer.tests.run_all 2>/dev/null || echo "FlashInfer tests not configured"
```

## Step 8: Run H100 Kernel Correctness Tests

```bash
# FlashMLA + DeepGEMM kernel correctness (requires SM90)
python3 -m glm5-kernels-flashmla-deepgemm.tests.run_all --h100
# Expected: All H100 tests pass (FlashMLA decode, sparse, FP8 KV; DeepGEMM logits, GEMM)

# FlashInfer kernel correctness
python3 -m glm5-kernels-flashinfer.tests.run_all --h100 2>/dev/null || echo "Run manually if available"
```

---

## Step 9: Run Benchmarks

### Quick Smoke Test (~5 minutes)

```bash
# Verify benchmark infrastructure works
cd /workspace/glm5
python3 -m benchmark.moe_sweep.bench_moe --quick --output-dir results/smoke/
python3 -m benchmark.mfu_ceiling.bench_mfu --quick --output-dir results/smoke/
```

### Individual Experiments

```bash
# ┌─────────────────────────────────────────────────────────────────────┐
# │ Experiment 1: MoE Sweep (SC '25 standard) — ~1 hour              │
# │ Sweeps: batch {1,16,32,64} × tokens {128,256,512,1024,2048}      │
# └─────────────────────────────────────────────────────────────────────┘
python3 -m benchmark.moe_sweep.bench_moe --output-dir results/moe/ 2>&1 | tee results/moe/log.txt

# ┌─────────────────────────────────────────────────────────────────────┐
# │ Experiment 2: Triple Report — Kernel Microbenchmarks — ~30 min    │
# │ All 9 components at full GLM-5 dims                                │
# └─────────────────────────────────────────────────────────────────────┘
python3 -m benchmark.triple_report.bench_micro --output-dir results/triple/ 2>&1 | tee results/triple/micro_log.txt

# ┌─────────────────────────────────────────────────────────────────────┐
# │ Experiment 3: Triple Report — Component Integration — ~30 min     │
# │ Full decoder layer (dense + MoE) across eager/flashmla/flashinfer  │
# └─────────────────────────────────────────────────────────────────────┘
python3 -m benchmark.triple_report.bench_component --output-dir results/triple/ 2>&1 | tee results/triple/component_log.txt

# ┌─────────────────────────────────────────────────────────────────────┐
# │ Experiment 4: Triple Report — End-to-End Serving — ~20 min        │
# │ TTFT + TPOT for chatbot, code_assist, long_doc_qa, agentic_swe    │
# └─────────────────────────────────────────────────────────────────────┘
python3 -m benchmark.triple_report.bench_e2e --output-dir results/triple/ 2>&1 | tee results/triple/e2e_log.txt

# ┌─────────────────────────────────────────────────────────────────────┐
# │ Experiment 5: MFU Ceiling Analysis — ~20 min                       │
# │ How close to FA3's 75% MFU at various (B, T, precision)           │
# └─────────────────────────────────────────────────────────────────────┘
python3 -m benchmark.mfu_ceiling.bench_mfu --output-dir results/mfu/ 2>&1 | tee results/mfu/log.txt

# ┌─────────────────────────────────────────────────────────────────────┐
# │ Experiment 6: FP8 Pareto Frontier — ~30 min                       │
# │ TFLOPS + cosine similarity for FlashMLA vs FlashInfer FP8 formats │
# └─────────────────────────────────────────────────────────────────────┘
python3 -m benchmark.fp8_pareto.bench_fp8 --output-dir results/fp8/ 2>&1 | tee results/fp8/log.txt

# ┌─────────────────────────────────────────────────────────────────────┐
# │ Experiment 7: Precision Experiment — ~10 min                       │
# │ Per-stage cosine similarity + RMSE across all implementations      │
# └─────────────────────────────────────────────────────────────────────┘
python3 -m benchmark.fp8_pareto.precision_experiment --output-dir results/precision/ 2>&1 | tee results/precision/log.txt

# ┌─────────────────────────────────────────────────────────────────────┐
# │ Experiment 8: Head-to-Head FlashMLA vs FlashInfer — ~3 hours      │
# │ Full parametric sweep: batch scaling, context scaling, memory      │
# └─────────────────────────────────────────────────────────────────────┘
python3 benchmark_head_to_head.py --experiment all --output-dir results/h2h/ 2>&1 | tee results/h2h/log.txt
```

### Run Everything (~5 hours total)

```bash
# Full benchmark suite orchestrator
python3 -m benchmark.run_all --output-dir results/full_run/ 2>&1 | tee results/full_run/log.txt

# Then the head-to-head comparison
python3 benchmark_head_to_head.py --experiment all --output-dir results/full_run/ 2>&1 | tee results/full_run/h2h_log.txt
```

### Quick Mode (~30 minutes total)

```bash
# Reduced sweeps for faster iteration
python3 -m benchmark.run_all --quick --output-dir results/quick/ 2>&1 | tee results/quick/log.txt
```

---

## Step 10: Profiling (Optional — Deep Analysis)

### nsys Timeline (Captures full CUDA timeline)

```bash
# Generate nsys profile for one forward pass
nsys profile \
  --trace=cuda,nvtx,osrt \
  --cuda-memory-usage=true \
  --output=results/nsys/glm5_profile \
  python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode nsys

# Download .nsys-rep file and open with: nsys-ui glm5_profile.nsys-rep
```

### ncu Kernel Metrics (Per-kernel analysis)

```bash
# Print the exact ncu commands to run
python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode commands

# Quick ncu profile (7 key metrics)
ncu --metrics \
  sm__throughput.avg.pct_of_peak_sustained_elapsed,\
  dram__throughput.avg.pct_of_peak_sustained_elapsed,\
  sm__pipe_tensor_op_hmma_cycles_active.avg.pct_of_peak_sustained_elapsed,\
  smsp__warps_issue_stalled_wait.pct,\
  dram__bytes_read.sum,\
  gpu__time_duration.sum,\
  sm__warps_active.avg.pct_of_peak_sustained_elapsed \
  --target-processes all \
  -o results/ncu/glm5_quick \
  python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench --mode ncu
```

### 3-Way Comparison (PyTorch vs Triton vs CUDA Kernels)

```bash
# Per-component comparison at full GLM-5 dims
python3 -m glm5-kernels-flashmla-deepgemm.tests.h100_bench_3way --full-dims 2>&1 | tee results/3way_log.txt
```

---

## Step 11: Collect Results

```bash
# All results are JSON files in results/
find results/ -name "*.json" | sort

# Quick summary of what was generated
echo "=== JSON Result Files ==="
find results/ -name "*.json" -exec echo {} \; -exec python3 -c "
import json, sys
with open(sys.argv[1]) as f:
    d = json.load(f)
print(f'  Experiment: {d.get(\"experiment\", \"?\")}')
print(f'  Results: {d.get(\"metadata\", {}).get(\"num_results\", \"?\")}')
print(f'  OOMs: {d.get(\"metadata\", {}).get(\"num_oom\", \"?\")}')
print()
" {} \;

# Download results to local machine (from your LOCAL terminal, not RunPod):
# scp -r runpod:/workspace/glm5/results/ ./glm5_results/
```

---

## Step 12: Unlock Clocks When Done

```bash
# Restore default dynamic clock management
sudo nvidia-smi -rgc  # Reset graphics clocks
sudo nvidia-smi -rpl  # Reset power limit
```

---

## Troubleshooting

### FlashMLA build fails
```bash
# Check CUDA toolkit version matches PyTorch
python3 -c "import torch; print(torch.version.cuda)"
nvcc --version
# Both must be 12.8+. If mismatch, install matching CUDA toolkit:
# apt-get install cuda-toolkit-12-8
```

### DeepGEMM JIT is slow (10-60s first call)
```bash
# Pre-warm the JIT cache
DG_JIT_USE_NVRTC=1 python3 -c "
import torch, deep_gemm
# Trigger JIT for common shapes
x = torch.randn(32, 6144, dtype=torch.float8_e4m3fn, device='cuda')
y = torch.randn(6144, 2048, dtype=torch.float8_e4m3fn, device='cuda')
print('JIT warming up...')
deep_gemm.fp8_gemm_nt(x, (y, torch.ones(1, device='cuda')), torch.empty(32, 2048, dtype=torch.bfloat16, device='cuda'))
print('JIT cache warmed')
"
# Subsequent runs will be instant (cached at ~/.deep_gemm/)
```

### FlashInfer qk_nope_head_dim=128 validation error
```bash
# GLM-5 has qk_nope_head_dim=192, FlashInfer's trtllm-gen validates for 128
# Apply the monkey-patch before importing:
python3 -c "
import flashinfer.mla as _mla
_orig = _mla._check_trtllm_gen_mla_shape
def _patched(*args):
    args = list(args)
    args[2] = 128  # Override qk_nope_head_dim validation
    return _orig(*args)
_mla._check_trtllm_gen_mla_shape = _patched
print('FlashInfer patched for GLM-5 dims')
"
```

### OOM during benchmarks
```bash
# Reduce batch size
python3 -m benchmark.moe_sweep.bench_moe --batch 1 16 --output-dir results/small/

# Or run with specific smaller config
python3 benchmark_head_to_head.py --experiment component --batch 4 --context 1024
```

### Check GPU thermal state during long benchmarks
```bash
# Monitor temperature in a second terminal
watch -n 5 'nvidia-smi --query-gpu=temperature.gpu,clocks.current.sm,power.draw --format=csv,noheader'
# If temp > 80°C, clocks will throttle even when locked. Check cooling.
```

---

## Estimated Runtimes on 1× H100 80GB SXM5

| Experiment | Quick Mode | Full Mode | Notes |
|-----------|-----------|-----------|-------|
| CPU tests (29) | ~10s | ~10s | No GPU needed |
| H100 kernel tests (36) | ~2 min | ~2 min | Correctness only |
| MoE sweep | ~5 min | ~1 hour | 4×5×4×3×3×2 = 1440 configs (full) |
| Triple Report micro | ~5 min | ~30 min | 9 components × 3 impls × 2 modes |
| Triple Report component | ~5 min | ~30 min | 3 configs × 2 layer types × 3 impls |
| Triple Report e2e | ~5 min | ~20 min | 4 scenarios × 3 impls |
| MFU ceiling | ~5 min | ~20 min | 4 components × (B,T) sweeps |
| FP8 Pareto | ~5 min | ~30 min | 3 components × 4 contexts × 2 formats |
| Precision experiment | ~2 min | ~10 min | 4 impls × 10 probes × N layers |
| Head-to-head (all) | ~15 min | ~3 hours | 7 experiments × parametric sweeps |
| **Total** | **~30 min** | **~7 hours** | Single H100 |

---

## File Outputs

All results are saved as JSON in the output directory:

```
results/
  moe/
    moe_sweep_20260323_143052.json          # Raw measurements + environment
  triple/
    triple_report_micro_20260323_143512.json
    triple_report_component_20260323_144023.json
    triple_report_e2e_20260323_144512.json
  mfu/
    mfu_ceiling_20260323_145023.json
  fp8/
    fp8_pareto_20260323_145523.json
  precision/
    precision_experiment_20260323_150023.json
  h2h/
    head_to_head_component_20260323_150523.json
    head_to_head_batch_scaling_20260323_152023.json
    head_to_head_context_scaling_20260323_153523.json
    ...
```

Each JSON includes:
- `environment`: GPU name, CUDA version, library versions, temperature, clock speed
- `results[]`: All 100 raw latency values per config + computed metrics (MFU, SOL%, TFLOPS)
- `metadata`: Experiment name, methodology reference (SC '25, FA3, MLPerf v5.1)
