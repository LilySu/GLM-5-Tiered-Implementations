# GLM-5

Two implementations of the GLM-5 model (arXiv 2602.15763v2), a 744B parameter
(40B active) Mixture-of-Experts language model from Zhipu AI.

```
glm5/
  glm5-raw-decoupled-from-hf/          Pure PyTorch reference (no HF dependency)
  glm5-triton/                          Self-contained model + Triton kernels from unsloth
  glm5-kernels-flashmla-deepgemm/       FlashMLA + DeepGEMM CUDA kernels (H100)
  glm5-kernels-flashinfer/              FlashInfer FA3 + DeepGEMM kernels (H100)
  benchmark/                            Academic benchmark suite (SC'25/MLSys/OSDI style)
  data/                                 Shared test data (both implementations use this)
  viz/                                  Architecture visualization
```

### Architecture visualization

Open [viz/glm5-architecture.html](viz/glm5-architecture.html) in a browser for a
3-column side-by-side comparison of **DeepSeek V3.2 vs GLM-5 Raw PyTorch vs
GLM-5 Triton**, with color-coded rows showing which components are Triton-
accelerated, which are shared between models, and what makes GLM-5 unique
(DSA, MLA, 256-expert MoE).

```bash
# Open in default browser (from WSL)
explorer.exe "$(wslpath -w viz/glm5-architecture.html)"

# Or on Mac/Linux
open viz/glm5-architecture.html
```

## glm5-raw-decoupled-from-hf/

The original standalone reimplementation, stripped of all HuggingFace
`transformers` abstractions. Single-file model (621 lines), dict-based config,
includes weight loading, tokenizer wrapper, and generation loop.

| File | Description |
|------|-------------|
| `model.py` | All model components in one file: RMSNorm, RoPE, DSAIndexer, MLAttention, FeedForward, TopkRouter, MoeExperts, MoE, DecoderLayer, GlmMoeDsaModel, GlmMoeDsaForCausalLM |
| `config.py` | Model configuration as a plain Python dict. Includes `load_config_from_hf()` to read HF checkpoints. |
| `cache.py` | KVCache replacing HuggingFace DynamicCache |
| `load_weights.py` | Loads multi-shard safetensors checkpoints |
| `generate.py` | Greedy, temperature, top-k, and top-p sampling |
| `tokenizer.py` | Thin wrapper around HF `tokenizers` library |
| `main.py` | Entry point: load -> tokenize -> generate |
| `test_standalone.py` | 8 unit tests |
| `verify.py` | Component-by-component shape verification |
| `validate.py` | Shared-data validation (uses `data/sample_data.py`) |

**Dependencies:** `torch`, `safetensors`, `tokenizers`

## glm5-triton/

Self-contained model that can run end-to-end with no imports outside the
directory. Includes Triton-accelerated kernels from
[unsloth](https://github.com/unslothai/unsloth) alongside the full PyTorch
model scaffolding.

Files prefixed with `unsloth_` are Triton kernels. Files without the prefix
are pure PyTorch.

```
+----------------------------------+------------------------+-------------------+
| Component                        | File                   | Status            |
+----------------------------------+------------------------+-------------------+
| RMSNorm fwd+bwd                  | unsloth_rms_layernorm  | Triton kernel     |
| SwiGLU fwd+bwd                   | unsloth_swiglu         | Triton kernel     |
| Cross-Entropy Loss (chunked)     | unsloth_cross_entropy  | Triton kernel     |
| LoRA MLP/QKV/W                   | unsloth_fast_lora      | Triton kernel     |
| MoE Grouped GEMM                 | unsloth_moe/           | Triton kernel     |
| Utilities                        | unsloth_utils          | Triton support    |
+----------------------------------+------------------------+-------------------+
| Config                           | config                 | PyTorch (dict)    |
| KV Cache                         | cache                  | PyTorch           |
| Decoupled partial-dim RoPE       | rope_partial           | PyTorch           |
| DSA Lightning Indexer            | dsa_indexer            | PyTorch           |
| DSA Sparse Attention             | dsa_sparse_attention   | PyTorch           |
| MLA (Multi-Latent Attention)     | mla_attention          | PyTorch           |
| FeedForward / MoE / Router       | model                  | PyTorch           |
| DecoderLayer / Base / CausalLM   | model                  | PyTorch           |
| MTP (Multi-Token Prediction)     | mtp                    | Stub              |
+----------------------------------+------------------------+-------------------+
```

**Dependencies:** `torch`, `triton`

See [glm5-triton/README.md](glm5-triton/README.md) for architecture details,
forward pass diagram, and the full Table 10 config from the paper.

## Shared test data

`data/sample_data.py` provides synthetic ChatML conversations used by both
implementations. No tokenizer or model weights needed.

```bash
# Inspect the sample data
python3 data/sample_data.py

# Run validation on glm5-triton (8 tests)
python3 glm5-triton/validate.py

# Run validation on the raw model (3 tests)
python3 glm5-raw-decoupled-from-hf/validate.py
```

The validation scripts build a tiny model (~395K params) with random weights,
feed structured ChatML token sequences through it, and verify:

| Test | What it proves |
|------|---------------|
| Forward pass | Logits shape correct, loss is finite |
| Backward pass | Gradients flow to 75%+ of parameters |
| Training convergence | Loss drops >80% in 20 steps on one batch |
| Label masking | `-100` tokens ignored, masked vs unmasked loss differs |
| KV cache decode | 10 autoregressive steps, cache grows by 1 each step |
| Multi-turn padded batch | Variable-length conversations, padded, loss decreases |
| Long sequence (256+ tok) | Attention + DSA indexer survive longer inputs |
| Gradient checkpointing | Same loss and gradients with/without checkpointing |

## Architecture

GLM-5 combines three innovations:

- **MLA (Multi-Latent Attention)** compresses KV into a 512-dim latent, applies
  RoPE to only a 64-dim decoupled stream. QK head dim = 192 (nope) + 64 (rope)
  = 256 total. V head dim = 256. 64 attention heads.

- **DSA (DeepSeek Sparse Attention)** uses a 32-head learned indexer to select
  the top-2048 most relevant tokens per query position, reducing attention cost
  by ~1.5-2x on long sequences.

- **256-Expert MoE** with sigmoid routing, group-based expert selection (top-8
  per token), 1 shared expert, and a 2.5x scaling factor. First 3 layers are
  dense SwiGLU, remaining 75 are sparse.

### Model config (Table 10 from paper)

```
+-------------------------+----------+
| Parameter               | GLM-5    |
+-------------------------+----------+
| Total Parameters        | 744B     |
| Activated Parameters    | 40B      |
| Dense Layers            | 3        |
| MoE Layers              | 75       |
| MTP Layers              | 1        |
| Hidden Dim              | 6144     |
| Dense Intermediate Dim  | 12288    |
| MoE Intermediate Dim    | 2048     |
| QK Head Dim (nope)      | 192      |
| QK Rope Head Dim        | 64       |
| V Head Dim              | 256      |
| Q LoRA Dim              | 2048     |
| KV LoRA Dim             | 512      |
| Attention Heads         | 64       |
| Indexer Attn Heads      | 32       |
| Indexer Head Dim        | 128      |
| Experts (total)         | 256      |
| Routed Experts (per tok)| 8        |
| Shared Experts          | 1        |
| Vocabulary Size         | 154880   |
+-------------------------+----------+
```

## Difference between the two directories

| | glm5-raw-decoupled-from-hf | glm5-triton |
|---|---|---|
| **Purpose** | Reference implementation, learning aid | Runnable model with GPU acceleration |
| **Model code** | Single 621-line `model.py` | Split across focused modules |
| **Triton kernels** | None | RMSNorm, SwiGLU, CE Loss, LoRA, MoE GEMM |
| **Weight loading** | Yes (`load_weights.py`) | No (config + model only) |
| **Generation** | Yes (`generate.py`) | No (forward pass only) |
| **Tokenizer** | Yes (`tokenizer.py`) | No |
| **Dependencies** | torch, safetensors, tokenizers | torch, triton |
| **Self-contained** | Yes (but needs checkpoint) | Yes (runs with random weights) |

## H100 Kernel Implementations

Two additional implementations use vendor CUDA kernels for H100 (SM90):

- **glm5-kernels-flashmla-deepgemm/** — FlashMLA for attention + DeepGEMM for MoE/DSA scoring
- **glm5-kernels-flashinfer/** — FlashInfer FA3 for attention + DeepGEMM for MoE/DSA scoring

See [benchmark/README.md](benchmark/README.md) for benchmark methodology and results.

## Docker Setup (RunPod / Cloud GPU)

For running on RunPod or any cloud GPU, see **[README-Docker-Setup.md](README-Docker-Setup.md)** — includes a pre-built Docker image with FlashMLA, DeepGEMM, FlashInfer, and PyTorch 2.8 so packages survive pod restarts.

## References

- [GLM-5 Technical Report](https://arxiv.org/abs/2602.15763) (arXiv 2602.15763v2)
- [Zhipu AI / zai-org](https://github.com/zai-org/GLM-5)
- [Unsloth](https://github.com/unslothai/unsloth) (Triton kernel source)

## License

See [LICENSE](LICENSE).
