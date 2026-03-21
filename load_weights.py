"""Load HuggingFace safetensors checkpoint into standalone GLM-MoE-DSA model."""

import glob
import re
import torch
from safetensors.torch import load_file


# Keys to skip (layer 78+ from some checkpoints, FP8 scale tensors)
_SKIP_RE = re.compile(r"model\.layers\.7[89]\.|model\.layers\.[89]\d\.|weight_scale_inv")


def assign(left, right, tensor_name="unknown"):
    """Copy right into left, checking shapes match."""
    if left.shape != right.shape:
        raise ValueError(
            f"Shape mismatch in '{tensor_name}'. "
            f"Model: {left.shape}, Checkpoint: {right.shape}"
        )
    with torch.no_grad():
        if isinstance(right, torch.Tensor):
            left.copy_(right)
        else:
            left.copy_(torch.as_tensor(right, dtype=left.dtype, device=left.device))
    return left


def load_weights(model, checkpoint_dir, device="cpu", dtype=torch.bfloat16):
    """Load safetensors weights from checkpoint_dir into a GlmMoeDsaForCausalLM.

    Handles:
    - Multi-shard safetensors files
    - Skipping layer 78+ and FP8 scale keys
    - Weight tying (lm_head ↔ embed_tokens)
    - Keeping indexer.weights_proj in fp32
    """
    # Find all safetensors shards
    shard_files = sorted(glob.glob(f"{checkpoint_dir}/*.safetensors"))
    if not shard_files:
        raise FileNotFoundError(f"No .safetensors files found in {checkpoint_dir}")

    # Merge all shards into one dict
    params = {}
    for shard in shard_files:
        params.update(load_file(shard, device=str(device)))

    # Get model state dict for reference
    state_dict = model.state_dict()
    loaded_keys = set()

    for ckpt_key, ckpt_tensor in params.items():
        # Skip unwanted keys
        if _SKIP_RE.search(ckpt_key):
            continue

        if ckpt_key in state_dict:
            assign(state_dict[ckpt_key], ckpt_tensor.to(state_dict[ckpt_key].dtype), ckpt_key)
            loaded_keys.add(ckpt_key)
        else:
            # Buffers (e.g. e_score_correction_bias) may not be in state_dict with strict mode
            # Try to set them directly on the model
            parts = ckpt_key.split(".")
            obj = model
            try:
                for part in parts[:-1]:
                    if part.isdigit():
                        obj = obj[int(part)]
                    else:
                        obj = getattr(obj, part)
                attr_name = parts[-1]
                target = getattr(obj, attr_name, None)
                if target is not None and isinstance(target, torch.Tensor):
                    assign(target, ckpt_tensor.to(target.dtype), ckpt_key)
                    loaded_keys.add(ckpt_key)
            except (AttributeError, IndexError):
                pass

    # Weight tying: if lm_head.weight not in checkpoint, copy from embed_tokens
    if "lm_head.weight" not in loaded_keys:
        if "model.embed_tokens.weight" in loaded_keys:
            model.lm_head.weight = model.model.embed_tokens.weight
            print("Weight tying: lm_head.weight ← model.embed_tokens.weight")

    # Move to target dtype, then restore fp32 for precision-sensitive modules
    model.to(dtype=dtype, device=device)
    for name, module in model.named_modules():
        if name.endswith("indexer.weights_proj"):
            module.weight.data = module.weight.data.to(torch.float32)

    # Report
    model_keys = set(state_dict.keys())
    missing = model_keys - loaded_keys
    unexpected = loaded_keys - model_keys
    if missing:
        print(f"Missing keys ({len(missing)}): {sorted(missing)[:10]}...")
    if unexpected:
        print(f"Unexpected keys ({len(unexpected)}): {sorted(unexpected)[:10]}...")
    print(f"Loaded {len(loaded_keys)} / {len(model_keys)} keys from {len(shard_files)} shard(s)")

    return model
