"""Standalone KV cache replacing HuggingFace DynamicCache."""

import torch


class KVCache:
    """Simple dynamic KV cache for autoregressive generation and training.

    Stores key/value tensors in [B, H, T, D] format per layer,
    concatenating new entries along the sequence dimension (dim=2).
    """

    def __init__(self, num_layers: int):
        self._cache: list[tuple[torch.Tensor, torch.Tensor] | None] = [None] * num_layers

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        layer_idx: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cache[layer_idx] is not None:
            prev_k, prev_v = self._cache[layer_idx]
            key_states = torch.cat([prev_k, key_states], dim=2)
            value_states = torch.cat([prev_v, value_states], dim=2)
        self._cache[layer_idx] = (key_states, value_states)
        return key_states, value_states

    def get_seq_length(self, layer_idx: int = 0) -> int:
        if self._cache[layer_idx] is None:
            return 0
        return self._cache[layer_idx][0].shape[2]

    def reset(self):
        self._cache = [None] * len(self._cache)
