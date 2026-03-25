"""MLA Roofline Decomposition Benchmark.

First published sub-operation roofline for Multi-head Latent Attention (MLA)
on H100. Decomposes MLA into 6 sub-operations and plots each on the roofline
to identify the bottleneck at each context length.

Sub-operations:
  1. q_a_proj:  hidden → q_lora_rank (compression)
  2. q_b_proj:  q_lora_rank → H * qk_head_dim (expansion)
  3. kv_a_proj: hidden → kv_lora_rank + rope_dim (compression)
  4. kv_b_proj: kv_lora_rank → H * (qk_nope + v_head) (expansion)
  5. attn:      Q×K^T + softmax + attn×V (core attention)
  6. o_proj:    H * v_head → hidden (output projection)

For comparison, also benchmarks standard MHA (no compression) to show
where MLA's compression trades compute for bandwidth.
"""
