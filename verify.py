"""GLM-MoE-DSA component verification — minimal, no test framework."""

import torch
from config import GLM_MOE_DSA_CONFIG
from cache import KVCache
from model import (
    RMSNorm,
    RotaryEmbedding,
    DSAIndexer,
    MLAttention,
    FeedForward,
    TopkRouter,
    MoeExperts,
    MoE,
    DecoderLayer,
    GlmMoeDsaModel,
    GlmMoeDsaForCausalLM,
)

# Small config for quick verification
SMALL_CONFIG = {
    "vocab_size": 1000,
    "hidden_size": 256,
    "intermediate_size": 512,
    "moe_intermediate_size": 128,
    "num_hidden_layers": 4,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_shared_experts": 1,
    "n_routed_experts": 8,
    "routed_scaling_factor": 2.5,
    "kv_lora_rank": 64,
    "q_lora_rank": 128,
    "qk_rope_head_dim": 16,
    "v_head_dim": 64,
    "qk_nope_head_dim": 48,
    "qk_head_dim": 64,  # 48 + 16
    "n_group": 1,
    "topk_group": 1,
    "num_experts_per_tok": 2,
    "norm_topk_prob": True,
    "hidden_act": "silu",
    "max_position_embeddings": 512,
    "initializer_range": 0.02,
    "rms_norm_eps": 1e-5,
    "attention_bias": False,
    "attention_dropout": 0.0,
    "index_topk": 32,
    "index_head_dim": 32,
    "index_n_heads": 4,
    "rope_theta": 10000.0,
    "dtype": "bfloat16",
    "mlp_layer_types": ["dense", "dense", "dense", "sparse"],
    "pad_token_id": None,
    "tie_word_embeddings": False,
}

cfg = SMALL_CONFIG
B, S = 2, 16

print("=" * 60)
print("GLM-MoE-DSA Component Verification")
print("=" * 60)

# 1. RMSNorm
print("\n[1] RMSNorm")
norm = RMSNorm(cfg["hidden_size"], eps=cfg["rms_norm_eps"])
x = torch.randn(B, S, cfg["hidden_size"])
out = norm(x)
print(f"    Input: {x.shape} -> Output: {out.shape}")
print(f"    Params: {sum(p.numel() for p in norm.parameters()):,}")

# 2. RotaryEmbedding
print("\n[2] RotaryEmbedding")
rope = RotaryEmbedding(cfg)
pos_ids = torch.arange(S).unsqueeze(0).expand(B, -1)
cos, sin = rope(x, pos_ids)
print(f"    cos: {cos.shape}, sin: {sin.shape}")

# 3. DSAIndexer
print("\n[3] DSAIndexer")
indexer = DSAIndexer(cfg, layer_idx=0)
hidden = torch.randn(B, S, cfg["hidden_size"])
q_resid = torch.randn(B, S, cfg["q_lora_rank"])
topk_indices = indexer(hidden, q_resid, (cos, sin), attention_mask=None)
print(f"    topk_indices: {topk_indices.shape}")
print(f"    Params: {sum(p.numel() for p in indexer.parameters()):,}")

# 4. MLAttention
print("\n[4] MLAttention")
attn = MLAttention(cfg, layer_idx=0)
attn_out, _ = attn(hidden, (cos, sin), attention_mask=None)
print(f"    Input: {hidden.shape} -> Output: {attn_out.shape}")
print(f"    Params: {sum(p.numel() for p in attn.parameters()):,}")

# 5. FeedForward
print("\n[5] FeedForward")
ff = FeedForward(cfg)
ff_out = ff(hidden)
print(f"    Input: {hidden.shape} -> Output: {ff_out.shape}")
print(f"    Params: {sum(p.numel() for p in ff.parameters()):,}")

# 6. TopkRouter
print("\n[6] TopkRouter")
router = TopkRouter(cfg)
router_logits = router(hidden)
print(f"    Input: {hidden.shape} -> Logits: {router_logits.shape}")
print(f"    Params: {sum(p.numel() for p in router.parameters()):,}")

# 7. MoeExperts
print("\n[7] MoeExperts")
experts = MoeExperts(cfg)
top_k_index = torch.randint(0, cfg["n_routed_experts"], (B * S, cfg["num_experts_per_tok"]))
top_k_weights = torch.softmax(torch.randn(B * S, cfg["num_experts_per_tok"]), dim=-1)
expert_out = experts(hidden.view(-1, cfg["hidden_size"]), top_k_index, top_k_weights)
print(f"    Output: {expert_out.shape}")
print(f"    Params: {sum(p.numel() for p in experts.parameters()):,}")

# 8. MoE
print("\n[8] MoE")
moe = MoE(cfg)
moe_out = moe(hidden)
print(f"    Input: {hidden.shape} -> Output: {moe_out.shape}")
print(f"    Params: {sum(p.numel() for p in moe.parameters()):,}")

# 9. DecoderLayer (dense, layer 0)
print("\n[9] DecoderLayer (dense)")
layer0 = DecoderLayer(cfg, layer_idx=0)
layer_out = layer0(hidden, position_embeddings=(cos, sin))
print(f"    Input: {hidden.shape} -> Output: {layer_out.shape}")
print(f"    Params: {sum(p.numel() for p in layer0.parameters()):,}")

# 10. DecoderLayer (sparse, layer 3)
print("\n[10] DecoderLayer (sparse/MoE)")
layer3 = DecoderLayer(cfg, layer_idx=3)
layer_out = layer3(hidden, position_embeddings=(cos, sin))
print(f"    Input: {hidden.shape} -> Output: {layer_out.shape}")
print(f"    Params: {sum(p.numel() for p in layer3.parameters()):,}")

# 11. GlmMoeDsaModel
print("\n[11] GlmMoeDsaModel")
base_model = GlmMoeDsaModel(cfg)
input_ids = torch.randint(0, cfg["vocab_size"], (B, S))
hidden_states, _ = base_model(input_ids)
print(f"    input_ids: {input_ids.shape} -> hidden_states: {hidden_states.shape}")
print(f"    Params: {sum(p.numel() for p in base_model.parameters()):,}")

# 12. GlmMoeDsaForCausalLM
print("\n[12] GlmMoeDsaForCausalLM")
model = GlmMoeDsaForCausalLM(cfg)
loss, logits, _ = model(input_ids, labels=input_ids)
print(f"    input_ids: {input_ids.shape} -> logits: {logits.shape}")
print(f"    loss: {loss.item():.4f}")
print(f"    Params: {sum(p.numel() for p in model.parameters()):,}")

# Quick forward pass (notebook style)
print("\n" + "=" * 60)
print("Quick forward pass (notebook style)")
print("=" * 60)
out = model(torch.tensor([[1, 2, 3]]))
print(f"model(torch.tensor([[1, 2, 3]])) -> logits shape: {out[1].shape}")

total = sum(p.numel() for p in model.parameters())
print(f"\nTotal parameters: {total:,}")
print("\nAll components verified successfully!")
