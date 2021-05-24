from typing import Tuple, Callable

from jax import numpy as jnp
from flax import linen as nn
from einops import rearrange

from ..layers import PatchEmbedBlock, FFBlock


class MixerBlock(nn.Module):
    tokens_expand_ratio: float
    channels_expand_ratio: float
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = rearrange(x, "... l d -> ... d l")
        x = FFBlock(
            expand_ratio=self.tokens_expand_ratio,
            activation_fn=self.activation_fn,
            dtype=self.dtype,
        )(x, is_training=is_training)
        x = rearrange(x, "... d l -> ... l d")
        x = x + inputs

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = FFBlock(
            expand_ratio=self.channels_expand_ratio,
            activation_fn=self.activation_fn,
            dtype=self.dtype,
        )(y, is_training=is_training)
        output = x + y
        return output


class MLPMixer(nn.Module):
    output_dim: int
    num_layers: int
    embed_dim: int
    patch_shape: Tuple[int]
    tokens_expand_ratio: float = 0.5
    channels_expand_ratio: float = 4
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        x = PatchEmbedBlock(
            patch_shape=self.patch_shape,
            embed_dim=self.embed_dim,
            use_bias=True,
            dtype=self.dtype,
        )(inputs)

        for _ in range(self.num_layers):
            x = MixerBlock(
                tokens_expand_ratio=self.tokens_expand_ratio,
                channels_expand_ratio=self.channels_expand_ratio,
                activation_fn=self.activation_fn,
                dtype=self.dtype,
            )(x, is_training=is_training)

        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = jnp.mean(x, axis=1)
        output = nn.Dense(features=self.output_dim, dtype=self.dtype)(x)
        return output
