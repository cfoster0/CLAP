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

        to_qkv = nn.Dense(features = self.dim_head * h * 3, use_bias = False)
        to_out = nn.Dense(features = dim_in)

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

        to_intermediate = nn.Dense(features = dim_in * mult)
        to_out = nn.Dense(features = dim_in)

        x = to_intermediate(x)
        x = nn.gelu(x)
        x = to_out(x)
        return x

class CLAP(nn.Module):
    dim: int
    depth: int
    heads: int

    def setup(self):
        self.self_attns = [(Attention(dim = self.dim, heads = self.heads), FeedForward()) for _ in range(self.depth)]

    @nn.compact
    def __call__(self, x):
        for attn, ff in self.self_attns:
            x = attn(x) + x
            x = ff(x) + x
        return x
