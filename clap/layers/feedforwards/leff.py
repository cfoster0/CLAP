from functools import partial
from typing import Callable

from jax import numpy as jnp
from flax import linen as nn
from einops import rearrange


class LeFFBlock(nn.Module):
    expand_ratio: float = None
    hidden_ch: int = None
    kernel_size: int = 5
    activation_fn: Callable = nn.activation.gelu
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):

        cls, tokens = inputs[:, 0], inputs[:, 1:]
        _, l, in_ch = tokens.shape
        if self.expand_ratio is None:
            if self.hidden_ch is None:
                raise ValueError(
                    'Must provide one of expand_ratio or hidden_ch')
            hidden_ch = self.hidden_ch
        else:
            hidden_ch = max(1, int(self.expand_ratio * in_ch))

        dense = partial(nn.Dense, use_bias=True, dtype=self.dtype)

        batch_norm = partial(nn.BatchNorm,
                             use_running_average=not is_training,
                             momentum=self.bn_momentum,
                             epsilon=self.bn_epsilon,
                             dtype=self.dtype)

        x = dense(features=hidden_ch)(tokens)
        x = batch_norm()(x)
        x = self.activation_fn(x)

        spatial_ch = int(jnp.sqrt(l))
        x = rearrange(x, 'b (h w) c -> b h w c', h=spatial_ch, w=spatial_ch)

        x = nn.Conv(
            features=hidden_ch,
            kernel_size=(self.kernel_size, self.kernel_size),
            padding='SAME',
            dtype=self.dtype,
        )(x)
        x = batch_norm()(x)
        x = self.activation_fn(x)

        x = rearrange(x, 'b h w c -> b (h w) c')

        x = dense(features=in_ch)(x)
        x = batch_norm()(x)
        x = self.activation_fn(x)

        cls_token = jnp.expand_dims(cls, axis=1)
        output = jnp.concatenate([cls_token, x], axis=1)
        return output
