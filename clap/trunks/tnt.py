from typing import Tuple, Optional, Callable

import flax.linen as nn
import jax.numpy as jnp
from einops import rearrange

from ..layers import AddAbsPosEmbed, SelfAttentionBlock, FFBlock, PatchEmbedBlock


class PixelEmbedBlock(nn.Module):
    patch_shape: Tuple[int]
    transformed_patch_shape: Tuple[int]
    embed_dim: int
    use_bias: bool = True
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs):
        assert self.patch_shape[0] % self.transformed_patch_shape[0] == 0
        assert self.patch_shape[1] % self.transformed_patch_shape[1] == 0

        x = rearrange(inputs,
                      'b (h p1) (w p2) c -> (b h w) p1 p2 c',
                      p1=self.patch_shape[0],
                      p2=self.patch_shape[0])
        x = rearrange(x,
                      'n (p1 t1) (p2 t2) c -> n (p1 p2) (c t1 t2)',
                      t1=self.transformed_patch_shape[0],
                      t2=self.transformed_patch_shape[1])
        output = nn.Dense(self.embed_dim,
                          use_bias=self.use_bias,
                          dtype=self.dtype)(x)
        return output


class Inner2OuterBlock(nn.Module):
    out_ch: Optional[int] = None
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, patch_inputs, pixel_inputs):
        b = patch_inputs.shape[0]
        out_ch = self.out_ch or patch_inputs.shape[-1]

        x = rearrange(pixel_inputs, '... n d -> ... (n d)')
        x = nn.Dense(features=out_ch, dtype=self.dtype)(x)
        x = rearrange(x, '(b l) d -> b l d', b=b)
        x = jnp.pad(x, ((0, 0), (1, 0), (0, 0)))
        output = x + patch_inputs
        return output


class EncoderBlock(nn.Module):
    inner_num_heads: int
    outer_num_heads: int
    inner_expand_ratio: float = 4
    outer_expand_ratio: float = 4
    attn_dropout_rate: float = 0.
    dropout_rate: float = 0.
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, patch_inputs, pixel_inputs, is_training: bool):
        inner_x = nn.LayerNorm(dtype=self.dtype)(pixel_inputs)
        inner_x = SelfAttentionBlock(num_heads=self.inner_num_heads,
                                     attn_dropout_rate=self.attn_dropout_rate,
                                     out_dropout_rate=self.dropout_rate,
                                     dtype=self.dtype)(inner_x,
                                                       is_training=is_training)
        inner_x = inner_x + pixel_inputs
        inner_y = nn.LayerNorm(dtype=self.dtype)(inner_x)
        inner_y = FFBlock(expand_ratio=self.inner_expand_ratio,
                          dropout_rate=self.dropout_rate,
                          dtype=self.dtype)(inner_y, is_training=is_training)
        inner_output = inner_x + inner_y

        outer_x = Inner2OuterBlock(dtype=self.dtype)(patch_inputs, inner_output)

        outer_x = nn.LayerNorm(dtype=self.dtype)(outer_x)
        outer_x = SelfAttentionBlock(num_heads=self.outer_num_heads,
                                     attn_dropout_rate=self.attn_dropout_rate,
                                     out_dropout_rate=self.dropout_rate,
                                     dtype=self.dtype)(outer_x,
                                                       is_training=is_training)
        outer_x = outer_x + patch_inputs
        outer_y = nn.LayerNorm(dtype=self.dtype)(outer_x)
        outer_y = FFBlock(expand_ratio=self.outer_expand_ratio,
                          dropout_rate=self.dropout_rate,
                          dtype=self.dtype)(outer_y, is_training=is_training)
        outer_output = outer_x + outer_y

        return outer_output, inner_output


class Encoder(nn.Module):
    num_layers: int
    inner_num_heads: int
    outer_num_heads: int
    inner_expand_ratio: float = 4
    outer_expand_ratio: float = 4
    attn_dropout_rate: float = 0.
    dropout_rate: float = 0.
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, patch_embeddings, pixel_embeddings, is_training: bool):
        for _ in range(self.num_layers):
            patch_embeddings, pixel_embeddings = EncoderBlock(
                inner_num_heads=self.inner_num_heads,
                outer_num_heads=self.outer_num_heads,
                attn_dropout_rate=self.attn_dropout_rate,
                dropout_rate=self.dropout_rate,
                activation_fn=self.activation_fn,
                dtype=self.dtype)(patch_embeddings,
                                  pixel_embeddings,
                                  is_training=is_training)

        output = patch_embeddings
        return output


class TNT(nn.Module):
    num_classes: int
    num_layers: int
    inner_num_heads: int
    outer_num_heads: int
    inner_embed_dim: int
    outer_embed_dim: int
    patch_shape: Tuple[int] = (16, 16)
    transformed_patch_shape: Tuple[int] = (4, 4)
    inner_expand_ratio: float = 4
    outer_expand_ratio: float = 4
    attn_dropout_rate: float = 0.
    dropout_rate: float = 0.
    activation_fn: Callable = nn.activation.gelu
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, is_training: bool):
        pixel_embeddings = PixelEmbedBlock(
            patch_shape=self.patch_shape,
            transformed_patch_shape=self.transformed_patch_shape,
            embed_dim=self.inner_embed_dim,
            dtype=self.dtype)(inputs)

        patch_embeddings = PatchEmbedBlock(patch_shape=self.patch_shape,
                                           embed_dim=self.outer_embed_dim,
                                           use_bias=True,
                                           dtype=self.dtype)(inputs)

        b, l, _ = patch_embeddings.shape
        cls_shape = (1, 1, self.outer_embed_dim)
        cls_token = self.param('cls', nn.initializers.zeros, cls_shape)
        cls_token = jnp.tile(cls_token, [b, 1, 1])
        patch_embeddings = jnp.concatenate([cls_token, patch_embeddings],
                                           axis=1)

        pixel_embeddings = AddAbsPosEmbed()(pixel_embeddings)
        patch_embeddings = AddAbsPosEmbed()(patch_embeddings)

        patch_embeddings = nn.Dropout(rate=self.dropout_rate)(
            patch_embeddings, deterministic=not is_training)

        patch_embeddings = Encoder(num_layers=self.num_layers,
                                   inner_num_heads=self.inner_num_heads,
                                   outer_num_heads=self.outer_num_heads,
                                   attn_dropout_rate=self.attn_dropout_rate,
                                   dropout_rate=self.dropout_rate,
                                   activation_fn=self.activation_fn,
                                   dtype=self.dtype)(patch_embeddings,
                                                     pixel_embeddings,
                                                     is_training=is_training)

        cls_token = patch_embeddings[:, 0]
        output = nn.Dense(
            features=self.num_classes,
            dtype=self.dtype,
            kernel_init=nn.initializers.zeros,
        )(cls_token)
        return output
