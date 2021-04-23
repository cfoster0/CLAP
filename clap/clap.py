import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as np

# einsum and einops

from jax.numpy import einsum
from einops import rearrange

# flax

import flax
from flax.core import freeze, unfreeze
from flax import linen as nn

# config

from jax.config import config
config.enable_omnistaging() # Linen requires enabling omnistaging

# main class

class Attention(nn.Module):
    dim: int
    heads: int
    dim_head: int = 64

    @nn.compact
    def __call__(self, x):
        dim_in, h = x.shape[-1], self.heads
        scale = dim_in ** -0.5

        norm = nn.LayerNorm()
        to_qkv = nn.Dense(features = self.dim_head * h * 3, use_bias = False)
        to_out = nn.Dense(features = dim_in)

        x = norm(x)
        qkv = np.split(to_qkv(x), 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h = h), qkv)

        sim = einsum('h i d, h j d -> h i j', q, k) * scale
        attn = nn.softmax(sim, axis = -1)

        out = einsum('h i j, h j d -> h i d', attn, v)
        out = rearrange(out, 'h n d -> n (h d)')
        return to_out(out)

class FeedForward(nn.Module):
    mult: int = 4

    @nn.compact
    def __call__(self, x):
        dim_in, mult = x.shape[-1], self.mult

        norm = nn.LayerNorm()
        to_intermediate = nn.Dense(features = dim_in * mult)
        to_out = nn.Dense(features = dim_in)

        x = norm(x)
        x = to_intermediate(x)
        x = nn.gelu(x)
        x = to_out(x)
        return x

class Transformer(nn.Module):
    dim: int
    depth: int
    heads: int
    dim_head: int = 64

    def setup(self):
        self.layers = [(Attention(dim = self.dim, heads = self.heads, dim_head = self.dim_head), FeedForward()) for _ in range(self.depth)]

    @nn.compact
    def __call__(self, x):
        to_norm_out = nn.LayerNorm()

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        x = to_norm_out(x)
        return x

class CLAP(nn.Module):
    text_num_tokens: int
    text_dim: int
    text_depth: int
    text_heads: int

    audio_dim: int
    audio_depth: int
    audio_heads: int

    def setup(self):
        self.audio_encoder = Transformer(dim = self.audio_dim, depth = self.audio_depth, heads = self.audio_heads)
        self.text_encoder = Transformer(dim = self.text_dim, depth = self.text_depth, heads = self.text_heads)

    @nn.compact
    def __call__(self, text, audio):
        num_tokens, text_dim = self.text_num_tokens, self.text_dim

        to_text_tokens = nn.Embed(num_embeddings = num_tokens, features = text_dim)
        text = to_text_tokens(text)

        enc_text = self.text_encoder(text)
        enc_audio = self.audio_encoder(audio)
        return enc_text, enc_audio
