#!/bin/bash
# Run after every RunPod pod restart to reinstall kernel packages.
# Usage: source /workspace/GLM-5-Decoupled-From-HuggingFace/setup.sh

echo "Setting up kernel packages..."

# FlashMLA — editable install is fine (no JIT, no bundled headers)
cd /workspace/FlashMLA
git submodule update --init --recursive --depth=1 2>/dev/null
FLASH_MLA_DISABLE_SM100=1 pip install -e . --no-build-isolation -q 2>/dev/null
echo "FlashMLA installed."

# DeepGEMM — must be non-editable so CUTLASS headers are bundled in site-packages
cd /workspace/DeepGEMM
rm -rf deep_gemm.egg-info 2>/dev/null
git checkout . 2>/dev/null
git submodule update --init --recursive --depth=1 2>/dev/null
pip install --no-build-isolation . -q
rm -rf /root/.deep_gemm 2>/dev/null
rm -rf /workspace/.deep_gemm_cache/cache 2>/dev/null
echo "DeepGEMM installed."

# FlashInfer — source install, needs PYTHONPATH
export PYTHONPATH=/workspace/flashinfer/python:$PYTHONPATH

# Remove /workspace/DeepGEMM from PYTHONPATH if present (shadows site-packages)
export PYTHONPATH=$(echo $PYTHONPATH | tr ':' '\n' | grep -v '/workspace/DeepGEMM' | tr '\n' ':' | sed 's/:$//')

# DeepGEMM JIT settings
export DG_JIT_USE_NVRTC=1
export DG_JIT_CACHE_DIR=/workspace/.deep_gemm_cache
mkdir -p $DG_JIT_CACHE_DIR

# Verify
cd /workspace
python3 -c "import flash_mla; print('FlashMLA OK:', flash_mla.__version__)"
python3 -c "import deep_gemm; print('DeepGEMM OK:', deep_gemm.__version__, 'from', deep_gemm.__path__)"
python3 -c "import flashinfer; print('FlashInfer OK')"

echo "All kernels ready."
