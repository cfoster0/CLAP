from typing import Callable, Tuple

import jax.numpy as jnp
import flax.linen as nn

from ..layers import PatchEmbedBlock, AddAbsPosEmbed, AttentionBlock, SelfAttentionBlock
from ..layers import LayerScaleBlock, StochasticDepthBlock, FFBlock


class ClassSelfAttentionBlock(AttentionBlock):

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        inputs_q = jnp.expand_dims(inputs[:, 0, :], axis=1)
        return super().__call__(inputs_q, inputs, is_training=is_training)


class EncoderBlock(nn.Module):
    num_heads: int
    stoch_depth_rate: float
    layerscale_eps: float
    expand_ratio: float = 4
    attn_dropout_rate: float = 0.
    dropout_rate: float = 0.
    activation_fn: Callable = nn.activation.gelu
    rotary_qk: bool = False
    rotary_v: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        x = nn.LayerNorm(dtype=self.dtype)(inputs)
        x = SelfAttentionBlock(num_heads=self.num_heads,
                               talking_heads=True,
                               attn_dropout_rate=self.attn_dropout_rate,
                               out_dropout_rate=self.dropout_rate,
                               rotary_qk=self.rotary_qk,
                               rotary_v=self.rotary_v,
                               dtype=self.dtype)(x, is_training=is_training)
        x = LayerScaleBlock(eps=self.layerscale_eps,
                            dtype=self.dtype)(x, is_training=is_training)
        x = StochasticDepthBlock(drop_rate=self.stoch_depth_rate)(
            x, is_training=is_training)
        x = x + inputs

        y = nn.LayerNorm(dtype=self.dtype)(x)
        y = FFBlock(expand_ratio=self.expand_ratio,
                    dropout_rate=self.dropout_rate,
                    activation_fn=self.activation_fn,
                    dtype=self.dtype)(y, is_training=is_training)
        y = LayerScaleBlock(eps=self.layerscale_eps,
                            dtype=self.dtype)(y, is_training=is_training)
        y = StochasticDepthBlock(drop_rate=self.stoch_depth_rate)(
            y, is_training=is_training)

        output = x + y
        return output


class Encoder(nn.Module):
    num_layers: int
    num_heads: int
    stoch_depth_rate: float
    layerscale_eps: float
    expand_ratio: float = 4
    attn_dropout_rate: float = 0.
    dropout_rate: float = 0.
    activation_fn: Callable = nn.activation.gelu
    rotary_qk: bool = False
    rotary_v: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        x = AddAbsPosEmbed()(inputs)
        x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not is_training)

        for _ in range(self.num_layers):
            x = EncoderBlock(num_heads=self.num_heads,
                             expand_ratio=self.expand_ratio,
                             attn_dropout_rate=self.attn_dropout_rate,
                             dropout_rate=self.dropout_rate,
                             stoch_depth_rate=self.stoch_depth_rate,
                             layerscale_eps=self.layerscale_eps,
                             activation_fn=self.activation_fn,
                             rotary_qk=self.rotary_qk,
                             rotary_v=self.rotary_v,
                             dtype=self.dtype)(x, is_training=is_training)

        output = x
        return output


class CAEncoderBlock(nn.Module):
    num_heads: int
    stoch_depth_rate: float
    layerscale_eps: float
    expand_ratio: float = 4
    attn_dropout_rate: float = 0.
    dropout_rate: float = 0.
    activation_fn: Callable = nn.activation.gelu
    rotary_qk: bool = False
    rotary_v: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, cls_token, is_training: bool):
        x = jnp.concatenate([cls_token, inputs], axis=1)
        x = nn.LayerNorm(dtype=self.dtype)(x)
        x = ClassSelfAttentionBlock(num_heads=self.num_heads,
                                    attn_dropout_rate=self.attn_dropout_rate,
                                    out_dropout_rate=self.dropout_rate,
                                    dtype=self.dtype)(x,
                                                      is_training=is_training)
        x = LayerScaleBlock(eps=self.layerscale_eps,
                            dtype=self.dtype)(x, is_training=is_training)
        x = StochasticDepthBlock(drop_rate=self.stoch_depth_rate)(
            x, is_training=is_training)
        cls_token = cls_token + x

        y = nn.LayerNorm(dtype=self.dtype)(cls_token)
        y = FFBlock(expand_ratio=self.expand_ratio,
                    dropout_rate=self.dropout_rate,
                    activation_fn=self.activation_fn,
                    dtype=self.dtype)(y, is_training=is_training)
        y = LayerScaleBlock(eps=self.layerscale_eps,
                            dtype=self.dtype)(y, is_training=is_training)
        y = StochasticDepthBlock(drop_rate=self.stoch_depth_rate)(
            y, is_training=is_training)

        output = cls_token + y
        return output


class CaiT(nn.Module):
    output_dim: int
    num_layers: int
    num_layers_token_only: int
    num_heads: int
    embed_dim: int
    patch_shape: Tuple[int]
    stoch_depth_rate: float
    layerscale_eps: float
    expand_ratio: float = 4
    attn_dropout_rate: float = 0.
    dropout_rate: float = 0.
    activation_fn: Callable = nn.activation.gelu
    rotary_qk: bool = False
    rotary_v: bool = False
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):

        x = PatchEmbedBlock(patch_shape=self.patch_shape,
                            embed_dim=self.embed_dim,
                            dtype=self.dtype)(inputs)

        x = Encoder(num_layers=self.num_layers,
                    num_heads=self.num_heads,
                    expand_ratio=self.expand_ratio,
                    attn_dropout_rate=self.attn_dropout_rate,
                    dropout_rate=self.dropout_rate,
                    stoch_depth_rate=self.stoch_depth_rate,
                    layerscale_eps=self.layerscale_eps,
                    rotary_qk=self.rotary_qk,
                    rotary_v=self.rotary_v,
                    activation_fn=self.activation_fn)(x,
                                                      is_training=is_training)

        b = x.shape[0]
        cls_shape = (1, 1, self.embed_dim)
        cls_token = self.param('cls', nn.initializers.zeros, cls_shape)
        cls_token = jnp.tile(cls_token, [b, 1, 1])

        for _ in range(self.num_layers_token_only):
            cls_token = CAEncoderBlock(num_heads=self.num_heads,
                                       expand_ratio=self.expand_ratio,
                                       attn_dropout_rate=self.attn_dropout_rate,
                                       dropout_rate=self.dropout_rate,
                                       stoch_depth_rate=self.stoch_depth_rate,
                                       layerscale_eps=self.layerscale_eps,
                                       activation_fn=self.activation_fn,
                                       rotary_qk=self.rotary_qk,
                                       rotary_v=self.rotary_v,
                                       dtype=self.dtype)(
                                           x,
                                           cls_token,
                                           is_training=is_training)

        x = jnp.concatenate([cls_token, x], axis=1)
        x = nn.LayerNorm(dtype=self.dtype)(x)

        cls_token = x[:, 0]
        output = nn.Dense(features=self.output_dim,
                          use_bias=True,
                          dtype=self.dtype,
                          kernel_init=nn.initializers.zeros)(cls_token)
        return output
