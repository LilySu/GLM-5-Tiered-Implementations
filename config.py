"""GLM-MoE-DSA model configuration as a plain Python dict."""

import json
import os


GLM_MOE_DSA_CONFIG = {
    # Vocabulary & embeddings
    "vocab_size": 154880,
    "hidden_size": 6144,
    "tie_word_embeddings": False,
    # Layers
    "num_hidden_layers": 78,
    "intermediate_size": 12288,
    # Attention
    "num_attention_heads": 64,
    "num_key_value_heads": 64,
    "attention_bias": False,
    "attention_dropout": 0.0,
    # MLA (Multi-head Latent Attention) dimensions
    "q_lora_rank": 2048,
    "kv_lora_rank": 512,
    "qk_rope_head_dim": 64,
    "qk_nope_head_dim": 192,
    "qk_head_dim": 256,  # qk_nope_head_dim + qk_rope_head_dim
    "v_head_dim": 256,
    # MoE (Mixture of Experts)
    "n_routed_experts": 256,
    "n_shared_experts": 1,
    "num_experts_per_tok": 8,
    "moe_intermediate_size": 2048,
    "routed_scaling_factor": 2.5,
    "n_group": 1,
    "topk_group": 1,
    "norm_topk_prob": True,
    # DSA (Dynamic Sparse Attention) indexer
    "index_topk": 2048,
    "index_head_dim": 128,
    "index_n_heads": 32,
    # Activation & normalization
    "hidden_act": "silu",
    "rms_norm_eps": 1e-5,
    # Positional encoding
    "max_position_embeddings": 202752,
    "rope_theta": 10000.0,
    # Weight initialization
    "initializer_range": 0.02,
    # Special tokens
    "pad_token_id": None,
    "bos_token_id": 0,
    "eos_token_id": 1,
    # Cache
    "use_cache": True,
    # MLP layer pattern: first 3 dense, rest sparse
    "mlp_layer_types": ["dense"] * 3 + ["sparse"] * 75,
}


def load_config_from_hf(checkpoint_dir: str) -> dict:
    """Read config.json from a HuggingFace checkpoint and return a standalone config dict."""
    config_path = os.path.join(checkpoint_dir, "config.json")
    with open(config_path) as f:
        hf_config = json.load(f)

    config = dict(GLM_MOE_DSA_CONFIG)

    # Direct mappings (keys that match between HF and standalone)
    direct_keys = [
        "vocab_size", "hidden_size", "num_hidden_layers", "intermediate_size",
        "num_attention_heads", "num_key_value_heads", "attention_bias", "attention_dropout",
        "q_lora_rank", "kv_lora_rank", "qk_rope_head_dim", "qk_nope_head_dim", "v_head_dim",
        "n_routed_experts", "n_shared_experts", "num_experts_per_tok",
        "moe_intermediate_size", "routed_scaling_factor", "n_group", "topk_group",
        "norm_topk_prob", "hidden_act", "rms_norm_eps", "max_position_embeddings",
        "initializer_range", "pad_token_id", "bos_token_id", "eos_token_id",
        "tie_word_embeddings", "use_cache",
        "index_topk", "index_head_dim", "index_n_heads",
    ]
    for key in direct_keys:
        if key in hf_config:
            config[key] = hf_config[key]

    # Computed fields
    config["qk_head_dim"] = config["qk_nope_head_dim"] + config["qk_rope_head_dim"]

    # Extract rope_theta from nested rope_parameters
    rope_params = hf_config.get("rope_parameters")
    if isinstance(rope_params, dict) and "rope_theta" in rope_params:
        config["rope_theta"] = rope_params["rope_theta"]
    elif "rope_theta" in hf_config:
        config["rope_theta"] = hf_config["rope_theta"]

    # MLP layer types
    if "mlp_layer_types" in hf_config and hf_config["mlp_layer_types"] is not None:
        config["mlp_layer_types"] = hf_config["mlp_layer_types"]
    else:
        n = config["num_hidden_layers"]
        config["mlp_layer_types"] = ["dense"] * min(3, n) + ["sparse"] * (n - 3)

    return config
