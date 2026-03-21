"""Text generation for GLM-MoE-DSA."""

import torch
import torch.nn.functional as F


def generate(model, input_ids, max_new_tokens=100, temperature=1.0,
             top_k=None, top_p=None, eos_token_id=None):
    """Generate token IDs autoregressively.

    Returns the full sequence (prompt + generated tokens).
    """
    model.eval()
    past_key_values = None
    generated = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            # Only feed new tokens when using cache
            if past_key_values is not None:
                curr_input = generated[:, -1:]
            else:
                curr_input = generated

            loss, logits, past_key_values = model(
                input_ids=curr_input, past_key_values=past_key_values, use_cache=True,
            )

            next_logits = logits[:, -1, :]  # [B, vocab]

            # Temperature
            if temperature != 1.0:
                next_logits = next_logits / temperature

            # Top-k filtering
            if top_k is not None:
                topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")

            # Top-p (nucleus) filtering
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            # Sample or greedy
            if temperature == 1.0 and top_k is None and top_p is None:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            generated = torch.cat([generated, next_token], dim=-1)

            # EOS check
            if eos_token_id is not None:
                if isinstance(eos_token_id, int):
                    eos_token_id_list = [eos_token_id]
                else:
                    eos_token_id_list = eos_token_id
                if next_token.item() in eos_token_id_list:
                    break

    return generated


def generate_stream(model, input_ids, max_new_tokens=100, temperature=1.0,
                    top_k=None, top_p=None, eos_token_id=None):
    """Streaming generator — yields one token at a time."""
    model.eval()
    past_key_values = None
    token_ids = input_ids

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if past_key_values is not None:
                curr_input = token_ids[:, -1:]
            else:
                curr_input = token_ids

            loss, logits, past_key_values = model(
                input_ids=curr_input, past_key_values=past_key_values, use_cache=True,
            )

            next_logits = logits[:, -1, :]

            if temperature != 1.0:
                next_logits = next_logits / temperature

            if top_k is not None:
                topk_vals, _ = torch.topk(next_logits, top_k, dim=-1)
                next_logits[next_logits < topk_vals[:, -1:]] = float("-inf")

            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(next_logits, descending=True, dim=-1)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - F.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                next_logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)

            if temperature == 1.0 and top_k is None and top_p is None:
                next_token = next_logits.argmax(dim=-1, keepdim=True)
            else:
                probs = F.softmax(next_logits, dim=-1)
                next_token = torch.multinomial(probs, 1)

            if eos_token_id is not None:
                ids = [eos_token_id] if isinstance(eos_token_id, int) else eos_token_id
                if next_token.item() in ids:
                    break

            yield next_token

            token_ids = torch.cat([token_ids, next_token], dim=1)
