#!/bin/bash
# Run after every RunPod pod restart to reinstall kernel packages.
# Usage: source /workspace/GLM-5-Decoupled-From-HuggingFace/setup.sh

cd /workspace/FlashMLA && FLASH_MLA_DISABLE_SM100=1 pip install -e . --no-build-isolation -q 2>/dev/null
cd /workspace/DeepGEMM && pip install -e . --no-build-isolation -q 2>/dev/null
export PYTHONPATH=/workspace/flashinfer/python:$PYTHONPATH
export DG_JIT_USE_NVRTC=1
export DG_JIT_CACHE_DIR=/workspace/.deep_gemm_cache
mkdir -p $DG_JIT_CACHE_DIR

python3 -c "import flash_mla; print('FlashMLA OK:', flash_mla.__version__)"
python3 -c "import deep_gemm; print('DeepGEMM OK:', deep_gemm.__version__)"
python3 -c "import flashinfer; print('FlashInfer OK')"

echo "All kernels ready."
