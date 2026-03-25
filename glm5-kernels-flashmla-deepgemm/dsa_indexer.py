# DSA Lightning Indexer — DeepGEMM kernel-accelerated version.
#
# When DeepGEMM is available (SM90), uses fp8_mqa_logits CUDA kernel for scoring.
# The kernel computes: logits[i,j] = sum_h(ReLU(q[i,h,:] @ kv[j,:]) * weights[i,h])
# This exactly matches GLM-5's indexer formula (Section 2.1.1).
#
# The deterministic torch.topk is always used for selection (Section 3.2).
#
# When DeepGEMM is not available, falls back to PyTorch eager computation.
#
# Dependencies: pip install deep-gemm (build from source, CUDA 12.8+, SM90)
#
# Paper ref: GLM-5 (arXiv 2602.15763v2), Section 2.1.1

import torch
import torch.nn as nn
import torch.nn.functional as F
from .rope_partial import apply_rotary_pos_emb

try:
    import deep_gemm
    from deep_gemm.utils import per_token_cast_to_fp8
    DEEP_GEMM_AVAILABLE = True
except ImportError:
    DEEP_GEMM_AVAILABLE = False


class DSAIndexer(nn.Module):
    """Selects top-k tokens for sparse attention via lightweight scoring.

    Architecture (GLM-5 config):
        index_n_heads:    32  (lightweight scoring heads)
        index_head_dim:   128 (dim per scoring head)
        index_topk:       2048 (max tokens to attend to)

    When DeepGEMM is available, the scoring computation is accelerated:
        PyTorch:   einsum('bshd,btd->bsht') * scale -> ReLU -> weighted sum
        DeepGEMM:  fp8_mqa_logits(q_fp8, kv_fp8, weights, cu_seqlens) — fused kernel
    """

    def __init__(self, cfg, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx

        self.hidden_size = cfg["hidden_size"]
        self.n_heads = cfg["index_n_heads"]
        self.head_dim = cfg["index_head_dim"]
        self.qk_rope_head_dim = cfg["qk_rope_head_dim"]
        self.index_topk = cfg["index_topk"]
        self.q_lora_rank = cfg["q_lora_rank"]

        self.wq_b = nn.Linear(self.q_lora_rank, self.n_heads * self.head_dim, bias=False)
        self.wk = nn.Linear(self.hidden_size, self.head_dim, bias=False)
        self.k_norm = nn.LayerNorm(self.head_dim, eps=1e-6)
        self.weights_proj = nn.Linear(self.hidden_size, self.n_heads, bias=False)
        self.softmax_scale = self.head_dim ** -0.5

        self._cached_keys = None
        self.use_deepgemm = DEEP_GEMM_AVAILABLE
        self._deepgemm_verified = False

    @torch.no_grad()
    def forward(self, hidden_states, q_resid, position_embeddings, attention_mask=None, use_cache=False):
        """Returns top-k token indices [B, S, topk]."""
        batch_size, seq_len, _ = hidden_states.shape
        cos, sin = position_embeddings

        # --- Queries ---
        q = self.wq_b(q_resid)
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim)
        q_pe, q_nope = torch.split(q, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        q_pe = apply_rotary_pos_emb(q_pe, cos, sin, unsqueeze_dim=2)
        q = torch.cat([q_pe, q_nope], dim=-1)

        # --- Keys ---
        k = self.k_norm(self.wk(hidden_states))
        k_pe, k_nope = torch.split(k, [self.qk_rope_head_dim, self.head_dim - self.qk_rope_head_dim], dim=-1)
        k_pe = apply_rotary_pos_emb(k_pe.unsqueeze(2), cos, sin, unsqueeze_dim=2).squeeze(2)
        k = torch.cat([k_pe, k_nope], dim=-1)

        # --- Key cache ---
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
        weights = self.weights_proj(hidden_states).float() * (self.n_heads ** -0.5)

        if self.use_deepgemm and batch_size == 1:
            # DeepGEMM path: fused FP8 MQA logits kernel
            # fp8_mqa_logits expects: q [seq_len, n_heads, head_dim], kv [seq_len_kv, head_dim]
            # Note: fp8_mqa_logits JIT fails on CUDA 12.8 (needs 12.9). Fall back gracefully.
            try:
                index_scores = self._deepgemm_score(q, k_cached, weights)
                self._deepgemm_verified = True
            except RuntimeError:
                if not self._deepgemm_verified:
                    self.use_deepgemm = False
                    index_scores = None
                else:
                    raise
            if index_scores is None:
                scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
                scores = F.relu(scores)
                index_scores = torch.einsum("bsht,bsh->bst", scores, weights)
        else:
            # PyTorch fallback
            scores = torch.einsum("bshd,btd->bsht", q.float(), k_cached.float()) * self.softmax_scale
            scores = F.relu(scores)
            index_scores = torch.einsum("bsht,bsh->bst", scores, weights)

        if attention_mask is not None:
            index_scores = index_scores + attention_mask

        # --- Deterministic TopK (Section 3.2: non-deterministic topk degrades RL) ---
        total_len = index_scores.shape[-1]
        topk = min(self.index_topk, total_len)
        return index_scores.topk(topk, dim=-1).indices

    def _deepgemm_score(self, q, k_cached, weights):
        """Use DeepGEMM fp8_mqa_logits for fused scoring on H100.

        API confirmed by debug_all_kernels2.py (Approach D):
        - q: raw FP8 tensor [S, H, D] (NOT a tuple)
        - kv: tuple of (FP8 tensor [T, D], 1D scales [T])
        - weights: [S, H] float32
        - cu_k_start/end: [S] int32

        The kernel fuses: ReLU(q · k^T) * weights → logits [S, T]
        """
        # q: [1, S, H, D], k_cached: [1, T, D], weights: [1, S, H]
        q_3d = q.squeeze(0)         # [S, H, D]
        k_2d = k_cached.squeeze(0)  # [T, D]
        w_2d = weights.squeeze(0)   # [S, H]

        seq_len = q_3d.shape[0]
        seq_len_kv = k_2d.shape[0]

        # ── FP8 quantization ──────────────────────────────────────────
        # Q: raw FP8 tensor (NOT a tuple) — Approach D from debug
        q_fp8 = q_3d.to(torch.float8_e4m3fn)

        # KV: tuple of (FP8 tensor [T, D], 1D scales [T])
        # Use per_token_cast_to_fp8 with ue8m0=True (confirmed by fix_kernels_h100.py)
        kv_tuple = per_token_cast_to_fp8(k_2d, use_ue8m0=True)
        kv_tuple = (kv_tuple[0], kv_tuple[1].squeeze(-1))  # scales must be 1D [T]

        # ── Causal sequence ranges ────────────────────────────────────
        cu_k_start = torch.zeros(seq_len, dtype=torch.int32, device=q.device)
        if seq_len == 1:
            # Decode: single query sees all cached tokens [0, T)
            cu_k_end = torch.full((1,), seq_len_kv, dtype=torch.int32, device=q.device)
        else:
            # Prefill: query i sees tokens [0, i+1) (causal)
            cu_k_end = torch.arange(
                seq_len_kv - seq_len + 1, seq_len_kv + 1,
                dtype=torch.int32, device=q.device,
            )

        # ── Kernel call ───────────────────────────────────────────────
        logits = deep_gemm.fp8_mqa_logits(
            q_fp8, kv_tuple, w_2d,
            cu_k_start, cu_k_end,
        )

        return logits.unsqueeze(0)  # [1, S, T]
