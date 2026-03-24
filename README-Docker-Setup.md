# GLM-5 Docker Setup: GPU Kernel Environment

Pre-built container with FlashMLA, DeepGEMM, FlashInfer, and Triton — ready for H100 benchmarks on RunPod, Nebius, or local GPU.

## What's Inside

| Component | Version | Install Method |
|-----------|---------|---------------|
| CUDA | 12.8.1 | Base image |
| PyTorch | 2.8.0 (cu128) | Base image / `pip install` |
| Triton | >=3.1 | `pip install` |
| FlashMLA | latest (SM90) | Built from source |
| DeepGEMM | latest (FP8 GEMM) | Built from source |
| FlashInfer | latest (cu128) | `pip install` wheels |
| scipy, pandas, numpy, safetensors | latest | `pip install` |

GPU requirement: **SM90 (Hopper)** — H100, H200, H800.

## Dockerfiles

| File | Base Image | PyTorch | Use Case |
|------|-----------|---------|----------|
| `Dockerfile.h100` | `runpod/pytorch:1.0.2-cu1281-torch280` | **2.8** | **Recommended.** RunPod H100 with correct PyTorch version. |
| `Dockerfile` | `nvidia/cuda:12.8.1-devel-ubuntu24.04` | 2.8 | Anywhere (local, Nebius, generic cloud). Multi-stage build. |
| `Dockerfile.runpod` | `runpod/pytorch:2.6.0` + upgrade to 2.8 | 2.8 | RunPod with their Jupyter/SSH/monitoring infra. |

**Use `Dockerfile.h100`** unless you have a specific reason for the others. It matches the exact environment where all 62 tests passed.

## Step-by-Step: Deploy to RunPod

### 1. Build the image (one-time, ~20-30 min)

On your local machine (needs Docker with NVIDIA support):

```bash
docker build -t glm5-kernels:latest -f Dockerfile.h100 .
```

### 2. Push to Docker Hub (one-time)

```bash
# Login to Docker Hub (create account at hub.docker.com if needed)
docker login

# Tag and push
docker tag glm5-kernels:latest YOUR_DOCKERHUB_USERNAME/glm5-kernels:latest
docker push YOUR_DOCKERHUB_USERNAME/glm5-kernels:latest
```

### 3. Create a RunPod pod with your image

1. Go to RunPod → **Deploy** → pick **H100 SXM 80GB**
2. Under **Container Image**, replace the default with: `YOUR_DOCKERHUB_USERNAME/glm5-kernels:latest`
3. Set volume mount: `/workspace`
4. Start the pod

### 4. Run benchmarks

SSH into the pod, then:

```bash
cd /workspace
git clone <your-glm5-repo> GLM-5-Decoupled-From-HuggingFace
cd GLM-5-Decoupled-From-HuggingFace

# Verify everything works
python3 benchmark/fix_kernels_h100.py

# Run benchmarks
python3 -m benchmark.mfu_ceiling.bench_mfu --output-dir results/mfu/
python3 -m benchmark.fp8_pareto.bench_fp8 --output-dir results/fp8/
python3 -m benchmark.moe_sweep.bench_moe --quick --output-dir results/moe/
```

**No more reinstalling packages after pod pause/resume.** Everything is baked into the Docker image.

## Why This Exists

Without a custom Docker image, every time a RunPod pod restarts (pause/resume or cold start):
- Python packages installed via `pip` are **lost** (they live on the ephemeral container filesystem)
- Only `/workspace` persists across restarts
- You have to manually reinstall FlashMLA, DeepGEMM, set PYTHONPATH, etc.

With the Docker image, all packages are baked into the image layer. Pod restarts don't affect them.

## Alternative: Manual Setup (no Docker build)

If you can't build the Docker image locally, start a stock RunPod pod and run:

```bash
# On a RunPod H100 pod — run this after every pod restart
cd /workspace/FlashMLA && FLASH_MLA_DISABLE_SM100=1 pip install -e . --no-build-isolation
cd /workspace/DeepGEMM && pip install -e . --no-build-isolation

# FlashInfer (if built from source)
export PYTHONPATH=/workspace/flashinfer/python:$PYTHONPATH

# DeepGEMM JIT settings
export DG_JIT_USE_NVRTC=1
export DG_JIT_CACHE_DIR=/workspace/.deep_gemm_cache

# Verify
python3 -c "import flash_mla; print('FlashMLA OK')"
python3 -c "import deep_gemm; print('DeepGEMM OK')"
python3 -c "import flashinfer; print('FlashInfer OK')"
```

The `-e .` (editable install) uses the source already in `/workspace`, so it's fast (~1 min for FlashMLA, seconds for DeepGEMM). But you do have to run it after every restart.

## Local Development

### Build and run interactively

```bash
./docker-build.sh
./docker-run.sh
```

### Run a benchmark directly

```bash
./docker-run.sh python3 -m benchmark.run_all --quick
```

## Build Options

```bash
# Recommended: H100 build
docker build -t glm5-kernels:latest -f Dockerfile.h100 .

# RunPod variant (with Jupyter/SSH)
./docker-build.sh --runpod

# Build and push to Docker Hub
./docker-build.sh --push --registry your-dockerhub-user

# Build and push to GitHub Container Registry
./docker-build.sh --push --registry ghcr.io/your-user
```

## DeepGEMM JIT Cache

DeepGEMM compiles CUDA kernels at runtime via JIT. First invocation at a new matrix size takes 10-30 seconds. Compiled kernels are cached in `DG_JIT_CACHE_DIR` (`/workspace/.deep_gemm_cache`).

The `docker-run.sh` script mounts `~/.deep_gemm_cache` from the host so the cache persists across container restarts. On RunPod, the `/workspace` volume persists automatically.

## Verifying the Installation

Inside the container:

```bash
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'SM: {torch.cuda.get_device_capability(0)}')

import triton; print(f'Triton {triton.__version__}')
import flash_mla; print('FlashMLA OK')
import deep_gemm; print(f'DeepGEMM {deep_gemm.__version__}')
import flashinfer; print('FlashInfer OK')
"
```

Expected output on H100:
```
PyTorch 2.8.0+cu128
CUDA 12.8
GPU: NVIDIA H100 80GB HBM3
SM: (9, 0)
Triton 3.4.0
FlashMLA OK
DeepGEMM 2.3.0
FlashInfer OK
```

## Troubleshooting

**FlashMLA build fails with "sm100 requires NVCC 12.9"**
Set `FLASH_MLA_DISABLE_SM100=1` before the build. SM100 is Blackwell (B200) — you don't need it on H100.

**Packages missing after pod restart**
You're using a stock RunPod template, not the custom Docker image. Either rebuild with the Docker image, or run the manual setup commands above after each restart.

**FlashMLA build fails with "unsupported SM"**
You're not on SM90. FlashMLA dense decode requires H100/H200. Check `torch.cuda.get_device_capability()`.

**DeepGEMM JIT compilation hangs**
Set `DG_JIT_USE_NVRTC=1` (already set in the Dockerfile). If still slow, check that CUDA toolkit matches PyTorch's CUDA version.

**FlashInfer import error about CUDA version**
Ensure the wheel matches your CUDA version. For CUDA 12.8: `pip install flashinfer-python`.

**OOM during benchmarks**
Reduce batch size or context length. Add `--shm-size=16g` to docker run if using multi-GPU.
