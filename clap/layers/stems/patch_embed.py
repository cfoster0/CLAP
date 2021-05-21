from typing import Tuple

from flax import linen as nn
from jax import numpy as jnp
from einops import rearrange


class PatchEmbedBlock(nn.Module):

    patch_shape: Tuple[int]
    embed_dim: int
    use_bias: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, *unused_args, **unused_kwargs):
        ph, pw = self.patch_shape

        x = rearrange(inputs,
                      'b (h ph) (w pw) c -> b (h w) (ph pw c)',
                      ph=ph,
                      pw=pw)
        output = nn.Dense(features=self.embed_dim,
                          use_bias=self.use_bias,
                          dtype=self.dtype)(x)
        return output
