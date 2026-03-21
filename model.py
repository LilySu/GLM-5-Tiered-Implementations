"""GLM-MoE-DSA model — standalone pure-PyTorch implementation."""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# RMSNorm
# ---------------------------------------------------------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        # Parameter named "weight" to match HF checkpoint key (e.g. input_layernorm.weight)
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


# ---------------------------------------------------------------------------
# RoPE helpers
# ---------------------------------------------------------------------------

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(x, cos, sin, unsqueeze_dim=1):
    """Apply rotary position embedding to a single tensor.

    unsqueeze_dim=1 for [B, H, S, D] (BHSD), =2 for [B, S, H, D] (BSHD).
    """
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (x * cos) + (rotate_half(x) * sin)


# ---------------------------------------------------------------------------
# KV head expansion
# ---------------------------------------------------------------------------

def repeat_kv(x, n_rep):
    """Expand (batch, n_kv_heads, seq, dim) → (batch, n_heads, seq, dim)."""
    batch, n_kv_heads, slen, head_dim = x.shape
    if n_rep == 1:
        return x
    x = x[:, :, None, :, :].expand(batch, n_kv_heads, n_rep, slen, head_dim)
    return x.reshape(batch, n_kv_heads * n_rep, slen, head_dim)


# ---------------------------------------------------------------------------
# Eager attention forward
# ---------------------------------------------------------------------------

def eager_attention_forward(module, query, key, value, attention_mask, scaling, dropout=0.0, **kwargs):
    key = repeat_kv(key, module.num_key_value_groups)
    value = repeat_kv(value, module.num_key_value_groups)

    attn_weights = torch.matmul(query, key.transpose(2, 3)) * scaling
    if attention_mask is not None:
        attn_weights = attn_weights + attention_mask

    attn_weights = F.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    attn_weights = F.dropout(attn_weights, p=dropout, training=module.training)
    attn_output = torch.matmul(attn_weights, value)
    attn_output = attn_output.transpose(1, 2).contiguous()

    return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Causal mask
# ---------------------------------------------------------------------------

def make_causal_mask(seq_len, past_len, dtype, device):
    """Create [1, 1, seq_len, total_len] causal mask.

    Position i can attend to 0 .. past_len+i. Masked positions get dtype min (≈ -inf).
    """
    total_len = past_len + seq_len
    rows = torch.arange(seq_len, device=device).unsqueeze(1)
    cols = torch.arange(total_len, device=device).unsqueeze(0)
    causal = cols <= (rows + past_len)
    mask = torch.where(causal, 0.0, torch.finfo(dtype).min)
    return mask.to(dtype).unsqueeze(0).unsqueeze(0)  # [1, 1, S, T]


# ---------------------------------------------------------------------------
# Rotary Embedding
# ---------------------------------------------------------------------------

class RotaryEmbedding(nn.Module):
    """Precomputes and caches inv_freq; returns (cos, sin) per forward call."""

    def __init__(self, cfg, device=None):
        super().__init__()
        # In this model, head_dim for RoPE is qk_rope_head_dim (64)
        dim = cfg["qk_rope_head_dim"]
        base = cfg["rope_theta"]

        inv_freq = 1.0 / (
            base ** (torch.arange(0, dim, 2, dtype=torch.int64).to(device=device, dtype=torch.float) / dim)
        )
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self.attention_scaling = 1.0

    @torch.no_grad()
    def forward(self, x, position_ids):
        # x: [B, S, D] — only used for dtype/device
        # position_ids: [B, S]
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1).to(x.device)
        position_ids_expanded = position_ids[:, None, :].float()

        freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
        emb = torch.cat((freqs, freqs), dim=-1)
        cos = emb.cos() * self.attention_scaling
        sin = emb.sin() * self.attention_scaling

        return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)


# ---------------------------------------------------------------------------
# DSA Indexer (Dynamic Sparse Attention)
# ---------------------------------------------------------------------------

class DSAIndexer(nn.Module):
    """Selects top-k tokens for sparse attention via lightweight scoring."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = cfg["hidden_size"]
        self.n_heads = cfg["index_n_heads"]
        self.head_dim = cfg["index_head_dim"]
        self.qk_rope_head_dim = cfg["qk_rope_head_dim"]
        self.index_topk = cfg["index_topk"]
        self.q_lora_rank = cfg["q_lora_rank"]

        # Names match checkpoint: wq_b, wk, k_norm, weights_proj
        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

        # Own key cache (separate from the main KVCache)
        self._cached_keys = None

    @torch.no_grad()
    def forward(self, hidden_states, q_resid, position_embeddings, attention_mask=None, use_cache=False):
        """Returns top-k token indices [B, S, topk]."""
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        # --- Queries ---
        q = self.wq_b(q_resid)  # [B, S, H*D]
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)  # [B, S, H, D]
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)

        # --- Keys ---
        k = self.k_norm(self.wk(hidden_states))  # [B, S, D]
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        # --- Key cache (managed by indexer, not KVCache) ---
        # Reset on prefill to avoid stale keys / batch-size mismatch
        if seq_len > 1:
            self._cached_keys = None

        if use_cache:
            if self._cached_keys is not None:
                k_cached = torch.cat([self._cached_keys, k], dim=1)
            else:
                k_cached = k
            self._cached_keys = k_cached
        else:
            k_cached = k

        # --- Scoring ---
        weights = self.weights_proj(hidden_states).float() * (self.n_heads ** -0.5)  # [B, S, H]

        # q·k^T per head → [B, S, H, T], then weight and sum across heads → [B, S, T]
        scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
        scores = F.relu(scores)
        index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        total_len = index_scores.shape[-1]
        topk = min(self.index_topk, total_len)
        return index_scores.topk(topk, dim=-1).indices  # [B, S, topk]


# ---------------------------------------------------------------------------
# MLA Attention (Multi-head Latent Attention) with DSA
# ---------------------------------------------------------------------------

class MLAttention(nn.Module):
    """Multi-head Latent Attention with DSA indexer for sparse token selection."""

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.num_key_value_groups = cfg["num_attention_heads"] // cfg["num_key_value_heads"]
        self.attention_dropout = cfg["attention_dropout"]
        self.num_heads = cfg["num_attention_heads"]

        self.q_lora_rank = cfg["q_lora_rank"]
        self.qk_rope_head_dim = cfg["qk_rope_head_dim"]
        self.kv_lora_rank = cfg["kv_lora_rank"]
        self.v_head_dim = cfg["v_head_dim"]
        self.qk_nope_head_dim = cfg["qk_nope_head_dim"]
        self.qk_head_dim = cfg["qk_head_dim"]

        self.is_causal = True

        # Query projection (LoRA path)
        if self.q_lora_rank is None:
            self.q_proj = nn.Linear(cfg["hidden_size"], self.num_heads * self.qk_head_dim, bias=False)
        else:
            self.q_a_proj = nn.Linear(cfg["hidden_size"], cfg["q_lora_rank"], bias=cfg["attention_bias"])
            self.q_a_layernorm = RMSNorm(cfg["q_lora_rank"])
            self.q_b_proj = nn.Linear(cfg["q_lora_rank"], self.num_heads * self.qk_head_dim, bias=False)

        # KV projections (MLA compressed path)
        self.kv_a_proj_with_mqa = nn.Linear(
            cfg["hidden_size"], self.kv_lora_rank + self.qk_rope_head_dim, bias=cfg["attention_bias"],
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank)
        self.kv_b_proj = nn.Linear(
            self.kv_lora_rank, self.num_heads * (self.qk_nope_head_dim + self.v_head_dim), bias=False,
        )

        # Output projection
        self.o_proj = nn.Linear(self.num_heads * self.v_head_dim, cfg["hidden_size"], bias=cfg["attention_bias"])

        self.scaling = self.qk_head_dim ** -0.5

        self.indexer = DSAIndexer(cfg, layer_idx)

    def forward(self, hidden_states, position_embeddings, attention_mask=None, past_key_values=None, **kwargs):
        batch_size, seq_length = hidden_states.shape[:-1]
        cos, sin = position_embeddings

        # --- Query path ---
        if self.q_lora_rank is None:
            query_states = self.q_proj(hidden_states)
            q_resid = None
        else:
            q_resid = self.q_a_layernorm(self.q_a_proj(hidden_states))  # [B, S, q_lora_rank]
            query_states = self.q_b_proj(q_resid)
        query_states = query_states.view(batch_size, seq_length, -1, self.qk_head_dim).transpose(1, 2)
        # Split nope/rope, apply RoPE, recombine — [B, H, S, D]
        q_nope, q_pe = torch.split(query_states, [self.qk_nope_head_dim, self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=1)

        # --- KV path ---
        compressed_kv = self.kv_a_proj_with_mqa(hidden_states)  # [B, S, kv_rank + rope_D]
        k_compressed, k_pe = torch.split(compressed_kv, [self.kv_lora_rank, self.qk_rope_head_dim], dim=-1)
        k_compressed = self.kv_a_layernorm(k_compressed)

        kv_expanded = self.kv_b_proj(k_compressed)  # [B, S, H * (nope_D + v_D)]
        kv_expanded = kv_expanded.view(batch_size, seq_length, -1, self.qk_nope_head_dim + self.v_head_dim)
        k_nope, value_states = torch.split(kv_expanded, [self.qk_nope_head_dim, self.v_head_dim], dim=-1)
        k_nope = k_nope.transpose(1, 2)       # [B, H, S, nope_D]
        value_states = value_states.transpose(1, 2)  # [B, H, S, v_D]

        # RoPE on k_pe (single-head rope stream)
        k_pe = k_pe.view(batch_size, 1, seq_length, self.qk_rope_head_dim)
        k_pe = apply_rotary_pos_emb(k_pe, cos, sin, unsqueeze_dim=1)
        k_pe = k_pe.expand(-1, k_nope.shape[1], -1, -1)  # [B, H, S, rope_D]

        # Assemble full Q and K
        query_states = torch.cat([q_nope, q_pe], dim=-1)   # [B, H, S, qk_head_dim]
        key_states = torch.cat([k_nope, k_pe], dim=-1)     # [B, H, S, qk_head_dim]

        # Cache update
        if past_key_values is not None:
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx)

        # --- Indexer (DSA sparse mask) ---
        # Convert 4D mask [B, 1, S, T] → 3D [B, S, T] for indexer
        indexer_mask = (
            attention_mask[:, 0, :, :]
            if attention_mask is not None and attention_mask.dim() == 4
            else attention_mask.unsqueeze(1) if attention_mask is not None
            else None
        )
        topk_indices = self.indexer(
            hidden_states, q_resid, position_embeddings, indexer_mask,
            use_cache=past_key_values is not None,
        )  # [B, S, topk]

        # Build combined DSA + causal mask: -inf everywhere except top-k positions
        total_len = key_states.shape[2]
        index_mask = torch.full(
            (batch_size, seq_length, total_len), float("-inf"),
            device=hidden_states.device, dtype=query_states.dtype,
        )
        index_mask.scatter_(-1, topk_indices, 0.0)      # [B, S, T]
        index_mask = index_mask.unsqueeze(1)              # [B, 1, S, T]
        if attention_mask is not None and attention_mask.dim() == 4:
            causal_mask = attention_mask[..., :total_len]
            combined_mask = index_mask + causal_mask
        else:
            combined_mask = (
                attention_mask.masked_fill(index_mask == float("-inf"), float("-inf"))
                if attention_mask is not None else index_mask
            )

        # Eager attention (no flash attention dispatch)
        attn_output, attn_weights = eager_attention_forward(
            self, query_states, key_states, value_states, combined_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
        )

        attn_output = attn_output.reshape(batch_size, seq_length, -1).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights


# ---------------------------------------------------------------------------
# Feed-Forward (SwiGLU)
# ---------------------------------------------------------------------------

class FeedForward(nn.Module):
    def __init__(self, cfg, intermediate_size=None):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.intermediate_size = cfg["intermediate_size"] if intermediate_size is None else intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


# ---------------------------------------------------------------------------
# TopkRouter
# ---------------------------------------------------------------------------

class TopkRouter(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.hidden_size = cfg["hidden_size"]
        self.n_routed_experts = cfg["n_routed_experts"]

        self.weight = nn.Parameter(torch.empty(self.n_routed_experts, self.hidden_size))
        self.register_buffer("e_score_correction_bias", torch.zeros(self.n_routed_experts, dtype=torch.float32))

    def forward(self, x):
        x = x.view(-1, self.hidden_size)
        return F.linear(x.float(), self.weight.float())


# ---------------------------------------------------------------------------
# MoeExperts (collection of expert weights as 3D tensors)
# ---------------------------------------------------------------------------

class MoeExperts(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.num_experts = cfg["n_routed_experts"]
        self.hidden_dim = cfg["hidden_size"]
        self.intermediate_dim = cfg["moe_intermediate_size"]
        self.gate_up_proj = nn.Parameter(torch.empty(self.num_experts, 2 * self.intermediate_dim, self.hidden_dim))
        self.down_proj = nn.Parameter(torch.empty(self.num_experts, self.hidden_dim, self.intermediate_dim))

    def forward(self, hidden_states, top_k_index, top_k_weights):
        final_hidden_states = torch.zeros_like(hidden_states)
        with torch.no_grad():
            expert_mask = F.one_hot(top_k_index, num_classes=self.num_experts)
            expert_mask = expert_mask.permute(2, 1, 0)
            expert_hit = torch.greater(expert_mask.sum(dim=(-1, -2)), 0).nonzero()

        for expert_idx in expert_hit:
            expert_idx = expert_idx[0]
            if expert_idx == self.num_experts:
                continue
            top_k_pos, token_idx = torch.where(expert_mask[expert_idx])
            current_state = hidden_states[token_idx]
            gate, up = F.linear(current_state, self.gate_up_proj[expert_idx]).chunk(2, dim=-1)
            current_hidden_states = F.silu(gate) * up
            current_hidden_states = F.linear(current_hidden_states, self.down_proj[expert_idx])
            current_hidden_states = current_hidden_states * top_k_weights[token_idx, top_k_pos, None]
            final_hidden_states.index_add_(0, token_idx, current_hidden_states.to(final_hidden_states.dtype))

        return final_hidden_states


# ---------------------------------------------------------------------------
# MoE (routed experts + shared experts)
# ---------------------------------------------------------------------------

class MoE(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.experts = MoeExperts(cfg)
        self.gate = TopkRouter(cfg)
        self.shared_experts = FeedForward(
            cfg, intermediate_size=cfg["moe_intermediate_size"] * cfg["n_shared_experts"],
        )
        self.n_routed_experts = cfg["n_routed_experts"]
        self.n_group = cfg["n_group"]
        self.topk_group = cfg["topk_group"]
        self.norm_topk_prob = cfg["norm_topk_prob"]
        self.routed_scaling_factor = cfg["routed_scaling_factor"]
        self.top_k = cfg["num_experts_per_tok"]

    def route_tokens_to_experts(self, router_logits):
        router_logits = router_logits.sigmoid()
        router_logits_for_choice = router_logits + self.gate.e_score_correction_bias
        group_scores = (
            router_logits_for_choice.view(-1, self.n_group, self.n_routed_experts // self.n_group)
            .topk(2, dim=-1)[0]
            .sum(dim=-1)
        )
        group_idx = torch.topk(group_scores, k=self.topk_group, dim=-1, sorted=False)[1]
        group_mask = torch.zeros_like(group_scores)
        group_mask.scatter_(1, group_idx, 1)
        score_mask = (
            group_mask.unsqueeze(-1)
            .expand(-1, self.n_group, self.n_routed_experts // self.n_group)
            .reshape(-1, self.n_routed_experts)
        )
        scores_for_choice = router_logits_for_choice.masked_fill(~score_mask.bool(), 0.0)
        topk_indices = torch.topk(scores_for_choice, k=self.top_k, dim=-1, sorted=False)[1]
        topk_weights = router_logits.gather(1, topk_indices)
        if self.norm_topk_prob:
            topk_weights = topk_weights / (topk_weights.sum(dim=-1, keepdim=True) + 1e-20)
        topk_weights = topk_weights * self.routed_scaling_factor
        return topk_indices, topk_weights

    def forward(self, hidden_states):
        residuals = hidden_states
        orig_shape = hidden_states.shape
        router_logits = self.gate(hidden_states)
        topk_indices, topk_weights = self.route_tokens_to_experts(router_logits)
        hidden_states = hidden_states.view(-1, hidden_states.shape[-1])
        hidden_states = self.experts(hidden_states, topk_indices, topk_weights).view(*orig_shape)
        hidden_states = hidden_states + self.shared_experts(residuals)
        return hidden_states


# ---------------------------------------------------------------------------
# Decoder Layer
# ---------------------------------------------------------------------------

class DecoderLayer(nn.Module):
    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.self_attn = MLAttention(cfg, layer_idx)

        if cfg["mlp_layer_types"][layer_idx] == "sparse":
            self.mlp = MoE(cfg)
        else:
            self.mlp = FeedForward(cfg)

        self.input_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.post_attention_layernorm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.gradient_checkpointing = False

    def _forward(self, hidden_states, attention_mask, position_embeddings, past_key_values=None, **kwargs):
        # Pre-norm → attention → residual
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states, position_embeddings, attention_mask=attention_mask,
            past_key_values=past_key_values, **kwargs,
        )
        hidden_states = residual + hidden_states

        # Pre-norm → MLP → residual
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states

    def forward(self, hidden_states, attention_mask=None, position_embeddings=None, past_key_values=None, **kwargs):
        if self.gradient_checkpointing and self.training:
            return torch.utils.checkpoint.checkpoint(
                self._forward, hidden_states, attention_mask, position_embeddings,
                past_key_values, use_reentrant=False, **kwargs,
            )
        return self._forward(hidden_states, attention_mask, position_embeddings, past_key_values, **kwargs)


# ---------------------------------------------------------------------------
# GlmMoeDsaModel (base model)
# ---------------------------------------------------------------------------

class GlmMoeDsaModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.padding_idx = cfg["pad_token_id"]
        self.vocab_size = cfg["vocab_size"]

        self.embed_tokens = nn.Embedding(cfg["vocab_size"], cfg["hidden_size"], self.padding_idx)
        self.layers = nn.ModuleList(
            [DecoderLayer(cfg, layer_idx) for layer_idx in range(cfg["num_hidden_layers"])]
        )
        self.norm = RMSNorm(cfg["hidden_size"], cfg["rms_norm_eps"])
        self.rotary_emb = RotaryEmbedding(cfg)

        self._init_weights()

    def _init_weights(self):
        std = self.cfg["initializer_range"]
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=std)
            elif isinstance(module, TopkRouter):
                nn.init.normal_(module.weight, mean=0.0, std=std)
                nn.init.zeros_(module.e_score_correction_bias)
            elif isinstance(module, MoeExperts):
                nn.init.normal_(module.gate_up_proj, mean=0.0, std=std)
                nn.init.normal_(module.down_proj, mean=0.0, std=std)

    def set_gradient_checkpointing(self, enable=True):
        for layer in self.layers:
            layer.gradient_checkpointing = enable

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, use_cache=None, **kwargs):
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if use_cache and past_key_values is None:
            from cache import KVCache
            past_key_values = KVCache(self.cfg["num_hidden_layers"])

        if position_ids is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            position_ids = torch.arange(inputs_embeds.shape[1], device=inputs_embeds.device) + past_seen_tokens
            position_ids = position_ids.unsqueeze(0)

        causal_mask = make_causal_mask(
            seq_len=inputs_embeds.shape[1],
            past_len=past_key_values.get_seq_length() if past_key_values is not None else 0,
            dtype=inputs_embeds.dtype,
            device=inputs_embeds.device,
        )

        hidden_states = inputs_embeds
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        for layer in self.layers:
            hidden_states = layer(
                hidden_states, attention_mask=causal_mask,
                position_embeddings=position_embeddings,
                past_key_values=past_key_values, **kwargs,
            )

        hidden_states = self.norm(hidden_states)
        return hidden_states, past_key_values


# ---------------------------------------------------------------------------
# GlmMoeDsaForCausalLM (model + LM head)
# ---------------------------------------------------------------------------

class GlmMoeDsaForCausalLM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = GlmMoeDsaModel(cfg)
        self.vocab_size = cfg["vocab_size"]
        self.lm_head = nn.Linear(cfg["hidden_size"], cfg["vocab_size"], bias=False)

        # Optionally tie weights
        if cfg.get("tie_word_embeddings", False):
            self.lm_head.weight = self.model.embed_tokens.weight

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, inputs_embeds=None, labels=None,
                use_cache=None, **kwargs):
        hidden_states, past_key_values = self.model(
            input_ids=input_ids, attention_mask=attention_mask,
            position_ids=position_ids, past_key_values=past_key_values,
            inputs_embeds=inputs_embeds, use_cache=use_cache, **kwargs,
        )

        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size), labels.view(-1),
            )

        return loss, logits, past_key_values
