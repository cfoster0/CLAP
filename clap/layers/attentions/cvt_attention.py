from functools import partial
from typing import Optional, Tuple

from flax import linen as nn
from jax import numpy as jnp

from einops import rearrange

from . import TalkingHeadsBlock


class ConvProjectionBlock(nn.Module):
    out_ch: int
    kernel_size: int = 3
    strides: int = 1
    use_bias: bool = True
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        in_ch = inputs.shape[-1]

        conv = partial(nn.Conv, dtype=self.dtype)

        x = conv(
            features=in_ch,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.strides, self.strides),
            padding="SAME",
            feature_group_count=in_ch,
            use_bias=False,
        )(inputs)
        x = nn.BatchNorm(
            use_running_average=not is_training,
            momentum=self.bn_momentum,
            epsilon=self.bn_epsilon,
            dtype=self.dtype,
        )(x)
        output = conv(features=self.out_ch, kernel_size=(1, 1), use_bias=self.use_bias)(
            x
        )
        return output


class CvTAttentionBlock(nn.Module):
    num_heads: int
    head_ch: Optional[int] = None
    out_ch: Optional[int] = None
    talking_heads: bool = False
    attn_dropout_rate: float = 0.0
    out_dropout_rate: float = 0.0
    kernel_size: int = 3
    strides: Tuple[int] = (1, 2, 2)
    use_bias: bool = False
    bn_momentum: float = 0.9
    bn_epsilon: float = 1e-5
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs_q, inputs_kv, is_training: bool):
        assert inputs_q.ndim == 4
        assert inputs_kv.ndim == 4
        assert len(self.strides) == 3
        q_strides, k_strides, v_strides = self.strides

        in_ch = inputs_q.shape[-1]
        assert in_ch % self.num_heads == 0
        head_ch = self.head_ch or int(in_ch / self.num_heads)
        out_ch = self.out_ch or in_ch

        conv_proj = partial(
            ConvProjectionBlock,
            out_ch=self.num_heads * head_ch,
            kernel_size=self.kernel_size,
            use_bias=self.use_bias,
            bn_momentum=self.bn_momentum,
            bn_epsilon=self.bn_epsilon,
            dtype=self.dtype,
        )

        query = conv_proj(strides=q_strides)(inputs_q, is_training=is_training)
        key = conv_proj(strides=k_strides)(inputs_kv, is_training=is_training)
        value = conv_proj(strides=v_strides)(inputs_kv, is_training=is_training)

        query = rearrange(query, "b H W (h d) -> b (H W) h d", h=self.num_heads)
        key = rearrange(key, "b H W (h d) -> b (H W) h d", h=self.num_heads)
        value = rearrange(value, "b H W (h d) -> b (H W) h d", h=self.num_heads)

        query = query / jnp.sqrt(head_ch)

        attn_weights = jnp.einsum("... q h d, ... k h d -> ... h q k", query, key)

        if self.talking_heads:
            attn_weights = TalkingHeadsBlock(num_heads=self.num_heads)(attn_weights)

        attn_weights = nn.softmax(attn_weights)

        if self.talking_heads:
            attn_weights = TalkingHeadsBlock(num_heads=self.num_heads)(attn_weights)

        attn_weights = nn.Dropout(rate=self.attn_dropout_rate)(
            attn_weights, deterministic=not is_training
        )

        attn_scores = jnp.einsum(
            "... h q k, ... k h d -> ... q h d", attn_weights, value
        )

        output = nn.DenseGeneral(
            features=out_ch, axis=(-2, -1), use_bias=self.use_bias, dtype=self.dtype
        )(attn_scores)

        output = nn.Dropout(rate=self.out_dropout_rate)(
            output, deterministic=not is_training
        )
        return output


class CvTSelfAttentionBlock(CvTAttentionBlock):
    @nn.compact
    def __call__(self, inputs, is_training: bool):
        return super().__call__(inputs, inputs, is_training)
