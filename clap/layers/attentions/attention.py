from functools import partial
from typing import Optional

from flax import linen as nn
from jax import numpy as jnp

from models.layers.attentions import TalkingHeadsBlock


class AttentionBlock(nn.Module):
    num_heads: int
    head_ch: Optional[int] = None
    out_ch: Optional[int] = None
    talking_heads: bool = False
    attn_dropout_rate: float = 0.
    out_dropout_rate: float = 0.
    use_bias: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, is_training: bool):
        assert inputs_q.ndim == inputs_kv.ndim == 3

        in_ch = inputs_q.shape[-1]
        assert in_ch % self.num_heads == 0
        head_ch = self.head_ch or int(in_ch / self.num_heads)
        out_ch = self.out_ch or in_ch

        dense = partial(nn.DenseGeneral,
                        axis=-1,
                        features=(self.num_heads, head_ch),
                        use_bias=self.use_bias,
                        dtype=self.dtype)

        query = dense(name='queries')(inputs_q)
        key = dense(name='keys')(inputs_kv)
        value = dense(name='values')(inputs_kv)

        query = query / jnp.sqrt(head_ch)

        attn_weights = jnp.einsum('... q h d, ... k h d -> ... h q k', query,
                                  key)

        if self.talking_heads:
            attn_weights = TalkingHeadsBlock(
                num_heads=self.num_heads)(attn_weights)

        attn_weights = nn.softmax(attn_weights)

        if self.talking_heads:
            attn_weights = TalkingHeadsBlock(
                num_heads=self.num_heads)(attn_weights)

        attn_weights = nn.Dropout(rate=self.attn_dropout_rate)(
            attn_weights, deterministic=not is_training)

        attn_scores = jnp.einsum('... h q k, ... k h d -> ... q h d',
                                 attn_weights, value)

        output = nn.DenseGeneral(features=out_ch,
                                 axis=(-2, -1),
                                 use_bias=self.use_bias,
                                 dtype=self.dtype)(attn_scores)

        output = nn.Dropout(rate=self.out_dropout_rate)(
            output, deterministic=not is_training)
        return output


class SelfAttentionBlock(AttentionBlock):

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        return super().__call__(inputs, inputs, is_training=is_training)
