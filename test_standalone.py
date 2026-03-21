"""Tests for standalone GLM-MoE-DSA model."""

import sys
import torch

# Ensure glm5 directory is importable
sys.path.insert(0, "/home/lily/wsl_git/glm5")

from model import (
    RMSNorm, RotaryEmbedding, DSAIndexer, MLAttention,
    FeedForward, TopkRouter, MoeExperts, MoE,
    DecoderLayer, GlmMoeDsaModel, GlmMoeDsaForCausalLM,
    rotate_half, apply_rotary_pos_emb, repeat_kv, make_causal_mask,
)
from cache import KVCache


# Small config for testing (matches HF structure but tiny dimensions)
SMALL_CFG = {
    "vocab_size": 256,
    "hidden_size": 64,
    "num_hidden_layers": 3,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "q_lora_rank": 32,
    "qk_rope_head_dim": 8,
    "kv_lora_rank": 16,
    "v_head_dim": 16,
    "qk_nope_head_dim": 12,
    "qk_head_dim": 20,  # 12 + 8
    "attention_bias": False,
    "attention_dropout": 0.0,
    "index_n_heads": 2,
    "index_head_dim": 10,
    "index_topk": 4,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "n_routed_experts": 4,
    "n_shared_experts": 1,
    "num_experts_per_tok": 2,
    "routed_scaling_factor": 2.5,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    "rms_norm_eps": 1e-5,
    "max_position_embeddings": 128,
    "rope_theta": 10000.0,
    "pad_token_id": None,
    "initializer_range": 0.02,
    "tie_word_embeddings": False,
    # First layer dense, rest sparse
    "mlp_layer_types": ["dense", "sparse", "sparse"],
}


def test_smoke_instantiate():
    """Model instantiates and has non-zero parameter count."""
    model = GlmMoeDsaForCausalLM(SMALL_CFG)
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params > 0, "Model has no parameters"
    print(f"  params: {n_params:,}")


def test_forward_shapes():
    """Forward pass produces correct logit shapes."""
    model = GlmMoeDsaForCausalLM(SMALL_CFG)
    ids = torch.randint(0, 256, (2, 8))

    loss, logits, cache = model(input_ids=ids)

    assert logits.shape == (2, 8, 256), f"Expected (2, 8, 256), got {logits.shape}"
    assert loss is None, "Loss should be None when no labels provided"
    assert cache is None, "Cache should be None when use_cache not set"
    print(f"  logits: {logits.shape}")


def test_forward_with_labels():
    """Forward pass with labels produces scalar loss."""
    model = GlmMoeDsaForCausalLM(SMALL_CFG)
    ids = torch.randint(0, 256, (2, 8))
    labels = torch.randint(0, 256, (2, 8))

    loss, logits, cache = model(input_ids=ids, labels=labels)

    assert loss is not None, "Loss should not be None with labels"
    assert loss.dim() == 0, f"Loss should be scalar, got shape {loss.shape}"
    assert loss.item() > 0, f"Loss should be positive, got {loss.item()}"
    print(f"  loss: {loss.item():.4f}")


def test_kv_cache():
    """Two-step generation correctly uses and extends KV cache."""
    model = GlmMoeDsaForCausalLM(SMALL_CFG)
    model.eval()

    ids = torch.randint(0, 256, (1, 6))

    # Step 1: prefill
    with torch.no_grad():
        _, logits1, cache = model(input_ids=ids, use_cache=True)

    assert cache is not None, "Cache should be returned when use_cache=True"
    assert cache.get_seq_length() == 6, f"Cache should have 6 tokens, got {cache.get_seq_length()}"

    # Step 2: decode one token
    next_token = logits1[:, -1:, :].argmax(dim=-1)
    with torch.no_grad():
        _, logits2, cache = model(input_ids=next_token, past_key_values=cache, use_cache=True)

    assert cache.get_seq_length() == 7, f"Cache should have 7 tokens, got {cache.get_seq_length()}"
    assert logits2.shape == (1, 1, 256), f"Expected (1, 1, 256), got {logits2.shape}"
    print(f"  cache seq_length after 2 steps: {cache.get_seq_length()}")


def test_gradient_flow():
    """loss.backward() runs without error and produces gradients."""
    model = GlmMoeDsaForCausalLM(SMALL_CFG)
    ids = torch.randint(0, 256, (1, 8))
    labels = torch.randint(0, 256, (1, 8))

    loss, logits, _ = model(input_ids=ids, labels=labels)
    loss.backward()

    # Check that at least some parameters got gradients
    has_grad = sum(1 for p in model.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    total = sum(1 for p in model.parameters() if p.requires_grad)
    assert has_grad > 0, "No parameters received gradients"
    print(f"  {has_grad}/{total} parameters have non-zero gradients")


def test_causal_mask():
    """Causal mask has correct shape and values."""
    mask = make_causal_mask(seq_len=4, past_len=0, dtype=torch.float32, device="cpu")
    assert mask.shape == (1, 1, 4, 4), f"Expected (1,1,4,4), got {mask.shape}"

    # Upper triangle should be -inf (masked), diagonal and below should be 0
    for i in range(4):
        for j in range(4):
            val = mask[0, 0, i, j].item()
            if j <= i:
                assert val == 0.0, f"mask[{i},{j}] should be 0, got {val}"
            else:
                assert val < -1e30, f"mask[{i},{j}] should be -inf, got {val}"

    # With past_len
    mask2 = make_causal_mask(seq_len=2, past_len=3, dtype=torch.float32, device="cpu")
    assert mask2.shape == (1, 1, 2, 5), f"Expected (1,1,2,5), got {mask2.shape}"
    # Row 0 can attend to positions 0..3 (past_len + 0)
    assert mask2[0, 0, 0, 3].item() == 0.0
    assert mask2[0, 0, 0, 4].item() < -1e30
    print("  causal mask OK")


def test_parameter_names_match_hf():
    """Parameter names match HF checkpoint convention for weight loading."""
    model = GlmMoeDsaForCausalLM(SMALL_CFG)
    names = set(model.state_dict().keys())

    # Check critical HF checkpoint keys exist
    expected_prefixes = [
        "model.embed_tokens.weight",
        "model.norm.weight",
        "lm_head.weight",
        # Layer 0 (dense) attention
        "model.layers.0.self_attn.q_a_proj.weight",
        "model.layers.0.self_attn.q_a_layernorm.weight",
        "model.layers.0.self_attn.q_b_proj.weight",
        "model.layers.0.self_attn.kv_a_proj_with_mqa.weight",
        "model.layers.0.self_attn.kv_a_layernorm.weight",
        "model.layers.0.self_attn.kv_b_proj.weight",
        "model.layers.0.self_attn.o_proj.weight",
        # Indexer
        "model.layers.0.self_attn.indexer.wq_b.weight",
        "model.layers.0.self_attn.indexer.wk.weight",
        "model.layers.0.self_attn.indexer.k_norm.weight",
        "model.layers.0.self_attn.indexer.k_norm.bias",
        "model.layers.0.self_attn.indexer.weights_proj.weight",
        # Norms
        "model.layers.0.input_layernorm.weight",
        "model.layers.0.post_attention_layernorm.weight",
        # Dense MLP (layer 0)
        "model.layers.0.mlp.gate_proj.weight",
        "model.layers.0.mlp.up_proj.weight",
        "model.layers.0.mlp.down_proj.weight",
        # Sparse MoE (layer 1)
        "model.layers.1.mlp.gate.weight",
        "model.layers.1.mlp.experts.gate_up_proj",
        "model.layers.1.mlp.experts.down_proj",
        "model.layers.1.mlp.shared_experts.gate_proj.weight",
        "model.layers.1.mlp.shared_experts.up_proj.weight",
        "model.layers.1.mlp.shared_experts.down_proj.weight",
    ]

    missing = []
    for key in expected_prefixes:
        if key not in names:
            missing.append(key)

    if missing:
        print(f"  WARNING: missing keys: {missing}")
        # Print all actual keys for debugging
        for n in sorted(names):
            print(f"    {n}")
        assert False, f"Missing {len(missing)} expected keys"
    else:
        print(f"  all {len(expected_prefixes)} expected keys present")


def test_components():
    """Individual components work correctly."""
    # RMSNorm
    norm = RMSNorm(64)
    x = torch.randn(2, 4, 64)
    out = norm(x)
    assert out.shape == x.shape
    print("  RMSNorm OK")

    # RotaryEmbedding
    rope = RotaryEmbedding(SMALL_CFG)
    pos = torch.arange(8).unsqueeze(0)
    cos, sin = rope(x[:, :1, :], pos)
    assert cos.shape == (1, 8, 8)  # [B, S, rope_dim]
    print("  RotaryEmbedding OK")

    # repeat_kv
    kv = torch.randn(1, 2, 4, 16)
    expanded = repeat_kv(kv, 3)
    assert expanded.shape == (1, 6, 4, 16)
    print("  repeat_kv OK")

    # KVCache
    cache = KVCache(2)
    k, v = torch.randn(1, 4, 3, 16), torch.randn(1, 4, 3, 16)
    k_out, v_out = cache.update(k, v, 0)
    assert cache.get_seq_length(0) == 3
    k2, v2 = torch.randn(1, 4, 1, 16), torch.randn(1, 4, 1, 16)
    k_out, v_out = cache.update(k2, v2, 0)
    assert cache.get_seq_length(0) == 4
    print("  KVCache OK")


# ---------------------------------------------------------------------------
# Run all tests
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("smoke_instantiate", test_smoke_instantiate),
        ("forward_shapes", test_forward_shapes),
        ("forward_with_labels", test_forward_with_labels),
        ("kv_cache", test_kv_cache),
        ("gradient_flow", test_gradient_flow),
        ("causal_mask", test_causal_mask),
        ("parameter_names_match_hf", test_parameter_names_match_hf),
        ("components", test_components),
    ]

    passed, failed = 0, 0
    for name, fn in tests:
        try:
            print(f"\n[TEST] {name}")
            fn()
            print(f"  PASSED")
            passed += 1
        except Exception as e:
            print(f"  FAILED: {e}")
            failed += 1

    print(f"\n{'=' * 40}")
    print(f"Results: {passed} passed, {failed} failed out of {len(tests)}")
    if failed > 0:
        sys.exit(1)
