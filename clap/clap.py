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

    kernel_init: Callable = nn.initializers.lecun_normal()
    bias_init: Callable = nn.initializers.zeros

    @nn.compact
    def __call__(self, x):
        inp_dim, h = x.shape[-1], self.heads
        scale = inp_dim ** -0.5

        qkv_W = self.param('qkv',
                           self.kernel_init,
                           (inp_dim, self.dim_head * h * 3))

        to_out = nn.Dense(features = inp_dim)

        qkv = np.split(x @ qkv_W, 3, axis = -1)
        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h = h), qkv)

        sim = einsum('h i d, h j d -> h i j', q, k) * scale
        attn = nn.softmax(sim, axis = -1)

        out = einsum('h i j, h j d -> h i d', attn, v)
        out = rearrange(out, 'h n d -> n (h d)')
        return to_out(out)

class CLAP(nn.Module):
    dim: int
    depth: int
    heads: int

    def setup(self):
        self.self_attns = [Attention(dim = self.dim, heads = self.heads) for _ in range(self.depth)]

    @nn.compact
    def __call__(self, x):
        for attn in self.self_attns:
            x = attn(x)
        return x
