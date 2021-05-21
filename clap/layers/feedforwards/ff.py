from functools import partial
from typing import Callable

from flax import linen as nn
from jax import numpy as jnp


class FFBlock(nn.Module):
    expand_ratio: float = None
    hidden_ch: int = None
    dropout_rate: float = 0.
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        in_ch = inputs.shape[-1]
        if self.expand_ratio is None:
            if self.hidden_ch is None:
                raise ValueError(
                    'Must provide one of expand_ratio or hidden_ch')
            hidden_ch = self.hidden_ch
        else:
            hidden_ch = max(1, int(self.expand_ratio * in_ch))

        dense = partial(nn.Dense, use_bias=True, dtype=self.dtype)

        x = dense(features=hidden_ch)(inputs)
        x = self.activation_fn(x)
        x = nn.Dropout(rate=self.dropout_rate, deterministic=not is_training)(x)
        x = dense(features=in_ch)(x)
        output = nn.Dropout(rate=self.dropout_rate,
                            deterministic=not is_training)(x)
        return output
