"""Cache-free Llama/SmolLM2 forward. Reuse weights, implement attention here.

One unpadded sequence on CPU, float32, inference only. No HF model.forward,
attention.forward, Cache, use_cache, or past_key_values is used in this model.
"""

import torch
from torch import nn
from transformers.models.llama.modeling_llama import apply_rotary_pos_emb


class Layer(nn.Module):
    def __init__(self, source):
        super().__init__()
        self.q = source.self_attn.q_proj
        self.k = source.self_attn.k_proj
        self.v = source.self_attn.v_proj
        self.out = source.self_attn.o_proj
        self.attention_norm = source.input_layernorm
        self.mlp_norm = source.post_attention_layernorm
        self.mlp = source.mlp


class Model(nn.Module):
    def __init__(self, source):
        super().__init__()
        config = source.config
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // self.heads
        self.embedding = source.model.embed_tokens
        self.layers = nn.ModuleList([Layer(layer) for layer in source.model.layers])
        self.rope = source.model.rotary_emb
        self.norm = source.model.norm
        self.lm_head = source.lm_head

    def forward(self, input_ids):
        x = self.embedding(input_ids)
        batch, length, width = x.shape
        positions = torch.arange(length, device=x.device)
        cos, sin = self.rope(x, positions.unsqueeze(0))
        allowed = positions[None, :] <= positions[:, None]

        for layer in self.layers:
            h = layer.attention_norm(x)
            q = layer.q(h).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
            k = layer.k(h).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
            v = layer.v(h).view(batch, length, self.kv_heads, self.head_dim).transpose(1, 2)
            q, k = apply_rotary_pos_emb(q, k, cos, sin)

            # Grouped-query attention: several query heads use each KV head.
            groups = self.heads // self.kv_heads
            keys = k.repeat_interleave(groups, dim=1)
            values = v.repeat_interleave(groups, dim=1)
            scores = (q @ keys.transpose(-2, -1)) * self.head_dim**-0.5
            scores = scores.masked_fill(~allowed, float("-inf"))
            attention = torch.softmax(scores, dim=-1) @ values
            attention = attention.transpose(1, 2).reshape(batch, length, width)
            x = x + layer.out(attention)
            x = x + layer.mlp(layer.mlp_norm(x))

        return self.lm_head(self.norm(x))
