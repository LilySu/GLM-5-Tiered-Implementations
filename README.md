# GLM-5 Standalone (Decoupled from HuggingFace Transformers)

A pure-PyTorch reimplementation of the **GLM-MoE-DSA** model architecture — Multi-head Latent Attention (MLA) + Dynamic Sparse Attention (DSA) + Mixture of Experts (MoE) — with zero dependency on the HuggingFace `transformers` library.

This repo follows the style of [LLMs-from-scratch](https://github.com/rasbt/LLMs-from-scratch) (specifically the [standalone Qwen3 notebook](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb)): self-contained `.py` files, dict-based config, minimal abstractions.

## Model

GLM-MoE-DSA is described in the [GLM-5 technical report](https://arxiv.org/abs/2501.00001) by [ZhipuAI](https://github.com/THUDM). The HuggingFace Transformers implementation lives at [`transformers/models/glm_moe_dsa`](https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm_moe_dsa).

**Architecture highlights:**
- **MLA** (Multi-head Latent Attention): LoRA-compressed queries and key-value projections, reducing KV cache size
- **DSA** (Dynamic Sparse Attention): learned indexer selects top-k tokens per position, achieving O(n·k) instead of O(n²)
- **MoE** (Mixture of Experts): 256 routed experts (8 active per token) + 1 shared expert; first 3 layers are dense, layers 4–78 are sparse
- **78 layers**, 6144 hidden dim, 154K vocab

| Spec | Value |
|------|-------|
| Parameters | ~600B (full model) |
| Layers | 78 (3 dense + 75 sparse) |
| Hidden dim | 6144 |
| Attention heads | 64 |
| Routed experts | 256 (top-8) |
| Context length | 202,752 |

## Files

| File | Description |
|------|-------------|
| `config.py` | Model configuration as a plain Python dict (`GLM_MOE_DSA_CONFIG`). Includes `load_config_from_hf()` to read a checkpoint's `config.json`. |
| `cache.py` | Standalone KV cache (`KVCache`) replacing HuggingFace's `DynamicCache`. Stores `[B, H, T, D]` tensors, concatenates along the sequence dimension. |
| `model.py` | All model components: `RMSNorm`, `RotaryEmbedding`, `DSAIndexer`, `MLAttention`, `FeedForward`, `TopkRouter`, `MoeExperts`, `MoE`, `DecoderLayer`, `GlmMoeDsaModel`, `GlmMoeDsaForCausalLM`. |
| `load_weights.py` | Loads multi-shard safetensors checkpoints into the standalone model. Handles weight tying, FP8 key skipping, layer 78+ filtering, and fp32 precision for the indexer. |
| `generate.py` | Text generation with greedy, temperature, top-k, and top-p (nucleus) sampling. Both batch (`generate`) and streaming (`generate_stream`) modes. |
| `tokenizer.py` | Thin wrapper around the HF `tokenizers` library (not `transformers`). Loads `tokenizer.json` from the checkpoint directory. |
| `main.py` | **Entry point.** Loads config → builds model → loads weights → tokenizes prompt → streams generated text. |
| `test_standalone.py` | Test suite: 8 tests covering instantiation, forward pass shapes, KV cache, gradient flow, causal mask, and HF checkpoint key compatibility. |
| `verify.py` | Component-by-component verification script — instantiates each module, runs a forward pass, prints shapes and parameter counts. |

## Module Name Mapping

| HuggingFace Transformers | This Repo | Notes |
|--------------------------|-----------|-------|
| `GlmMoeDsaRMSNorm` | `RMSNorm` | Removed `@use_kernel_forward_from_hub` decorator |
| `GlmMoeDsaRotaryEmbedding` | `RotaryEmbedding` | Inlined default RoPE; removed `ROPE_INIT_FUNCTIONS`, `@dynamic_rope_update` |
| `GlmMoeDsaIndexer` | `DSAIndexer` | Config access via dict instead of dataclass attributes |
| `GlmMoeDsaAttention` | `MLAttention` | Removed flash attention dispatch; calls `eager_attention_forward` directly |
| `GlmMoeDsaMLP` | `FeedForward` | `ACT2FN[config.hidden_act]` → `F.silu` |
| `GlmMoeDsaTopkRouter` | `TopkRouter` | Config via dict |
| `GlmMoeDsaNaiveMoe` | `MoeExperts` | Removed `@use_experts_implementation` decorator |
| `GlmMoeDsaMoE` | `MoE` | Config via dict |
| `GlmMoeDsaDecoderLayer` | `DecoderLayer` | `nn.Module` instead of `GradientCheckpointingLayer`; manual checkpoint support |
| `GlmMoeDsaModel` | `GlmMoeDsaModel` | `nn.Module` instead of `PreTrainedModel`; returns tuple |
| `GlmMoeDsaForCausalLM` | `GlmMoeDsaForCausalLM` | `nn.Module`; no `GenerationMixin`; `F.cross_entropy` for loss |

## Dependency Replacements

| HF Transformers Dependency | Standalone Replacement |
|---|---|
| `PreTrainedConfig` | Plain Python dict |
| `PreTrainedModel` | `nn.Module` with `_init_weights` |
| `Cache` / `DynamicCache` | `KVCache` (cache.py) |
| `GenerationMixin` | `generate()` / `generate_stream()` (generate.py) |
| `create_causal_mask` | `make_causal_mask()` inline function |
| `ALL_ATTENTION_FUNCTIONS` | Direct `eager_attention_forward` call |
| `ROPE_INIT_FUNCTIONS` / `@dynamic_rope_update` | Inlined default RoPE computation |
| `ACT2FN` | `F.silu` hardcoded |
| `@use_experts_implementation` | Removed (uses naive loop) |
| `@use_kernel_forward_from_hub` | Removed |
| `BaseModelOutputWithPast` / `CausalLMOutputWithPast` | Tuple returns |
| `@auto_docstring` / `@can_return_tuple` / `@capture_outputs` | Removed |
| `FlashAttentionKwargs` / `Unpack[TransformersKwargs]` | Removed |
| `initialization` module | `torch.nn.init` directly |

## Quick Start

### Prerequisites

```bash
pip install torch safetensors tokenizers
```

### Inference

Edit `CHECKPOINT_DIR` in `main.py` to point to a downloaded GLM-5 checkpoint, then:

```bash
python main.py
```

Or from a Python script:

```python
import torch
from config import load_config_from_hf
from model import GlmMoeDsaForCausalLM
from load_weights import load_weights
from tokenizer import GLMTokenizer
from generate import generate

cfg = load_config_from_hf("/path/to/checkpoint")
model = GlmMoeDsaForCausalLM(cfg)
model = load_weights(model, "/path/to/checkpoint", device="cuda", dtype=torch.bfloat16)
model.eval()

tokenizer = GLMTokenizer("/path/to/checkpoint/tokenizer.json")
ids = torch.tensor([tokenizer.encode("Hello")], device="cuda")
output = generate(model, ids, max_new_tokens=100, eos_token_id=tokenizer.eos_token_id)
print(tokenizer.decode(output[0].tolist()))
```

### Training

```python
import torch
from config import GLM_MOE_DSA_CONFIG
from model import GlmMoeDsaForCausalLM

cfg = GLM_MOE_DSA_CONFIG  # or load_config_from_hf(...)
model = GlmMoeDsaForCausalLM(cfg)
model.to("cuda")

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

for batch in dataloader:
    input_ids, labels = batch
    loss, logits, _ = model(input_ids=input_ids, labels=labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
```

To enable gradient checkpointing (reduces memory at the cost of compute):

```python
model.model.set_gradient_checkpointing(enable=True)
```

### Tests

```bash
python test_standalone.py   # 8 unit tests
python verify.py            # component-by-component verification with shapes and param counts
```

## References

- [GLM-5 Technical Report](https://arxiv.org/abs/2501.00001)
- [ZhipuAI (zai-org)](https://github.com/THUDM)
- [HuggingFace Transformers — glm_moe_dsa](https://github.com/huggingface/transformers/tree/main/src/transformers/models/glm_moe_dsa)
- [LLMs-from-scratch — Qwen3 standalone](https://github.com/rasbt/LLMs-from-scratch/blob/main/ch05/11_qwen3/standalone-qwen3.ipynb)

## License

See [LICENSE](LICENSE).
