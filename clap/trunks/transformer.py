from typing import Tuple, Callable

from flax import linen as nn
from jax import numpy as jnp

from ..layers import SelfAttentionBlock, FFBlock, AddAbsPosEmbed


class EncoderBlock(nn.Module):
    num_heads: int
    expand_ratio: float = 4
    attn_dropout_rate: float = 0.0
    dropout_rate: float = 0.0
    activation_fn: Callable = nn.activation.gelu
    rotary_qk: bool = False
    rotary_v: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = SelfAttentionBlock(
            num_heads=self.num_heads,
            attn_dropout_rate=self.attn_dropout_rate,
            out_dropout_rate=self.dropout_rate,
            rotary_qk=self.rotary_qk,
            rotary_v=self.rotary_v,
            dtype=self.dtype,
        )(x, is_training=is_training)
        x = x + inputs

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = FFBlock(
            expand_ratio=self.expand_ratio,
            dropout_rate=self.dropout_rate,
            activation_fn=self.activation_fn,
            dtype=self.dtype,
        )(y, is_training=is_training)
        output = x + y
        return output


class Encoder(nn.Module):
    num_layers: int
    num_heads: int
    expand_ratio: float = 4
    attn_dropout_rate: float = 0.0
    dropout_rate: float = 0.0
    activation_fn: Callable = nn.activation.gelu
    rotary_qk: bool = False
    rotary_v: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        if not self.rotary_qk and not self.rotary_v:
            x = AddAbsPosEmbed()(inputs)
        else:
            x = inputs
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)

        for _ in range(self.num_layers):
            x = EncoderBlock(
                num_heads=self.num_heads,
                expand_ratio=self.expand_ratio,
                attn_dropout_rate=self.attn_dropout_rate,
                dropout_rate=self.dropout_rate,
                activation_fn=self.activation_fn,
                rotary_qk=self.rotary_qk,
                rotary_v=self.rotary_v,
                dtype=self.dtype,
            )(x, is_training=is_training)

        output = nn.LayerNorm(dtype=self.dtype)(x)
        return output


class Transformer(nn.Module):
    output_dim: int
    num_layers: int
    num_heads: int
    embed_dim: int
    expand_ratio: float = 4
    attn_dropout_rate: float = 0.0
    dropout_rate: float = 0.0
    activation_fn: Callable = nn.activation.gelu
    rotary_qk: bool = False
    rotary_v: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        assert self.embed_dim % self.num_heads == 0

        x = inputs

        b, l, _ = x.shape
        cls_shape = (1, 1, self.embed_dim)
        cls_token = self.param("cls", nn.initializers.zeros, cls_shape)
        cls_token = jnp.tile(cls_token, [b, 1, 1])
        x = jnp.concatenate([cls_token, x], axis=1)

        x = Encoder(
            num_layers=self.num_layers,
            num_heads=self.num_heads,
            expand_ratio=self.expand_ratio,
            attn_dropout_rate=self.attn_dropout_rate,
            dropout_rate=self.dropout_rate,
            activation_fn=self.activation_fn,
            rotary_qk=self.rotary_qk,
            rotary_v=self.rotary_v,
            dtype=self.dtype,
        )(x, is_training=is_training)

        cls_token = x[:, 0]
        output = nn.Dense(
            features=self.output_dim,
            dtype=self.dtype,
            kernel_init=nn.initializers.zeros,
        )(cls_token)
        return output
