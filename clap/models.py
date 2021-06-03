import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp, vmap, jit
from jax.ops import index, index_update
from .trunks import Transformer, ViT, TNT, CaiT, MLPMixer

# einsum and einops

from jax.numpy import einsum
from einops import rearrange, repeat

# flax

import flax
from flax.core import freeze, unfreeze
from flax import linen as nn

# constants

LARGE_NEG_VALUE = -1e10

# config

from jax.config import config

config.enable_omnistaging()  # Linen requires enabling omnistaging

# helpers


def cross_entropy(logits, targets, axis=-1):
    logprobs = nn.log_softmax(logits, axis=axis)
    nll = jnp.take_along_axis(logprobs, jnp.expand_dims(targets, axis=axis), axis=axis)
    ce = -jnp.mean(nll)
    return ce


# main class


class CLAP(nn.Module):
    text_config: Any

    audio_config: Any

    temp_init: Callable = nn.initializers.zeros

    def setup(self):
        if self.text_config.kind == "transformer":
            self.text_encoder = Transformer(
                output_dim=self.text_config.projection_dim,
                num_layers=self.text_config.depth,
                num_heads=self.text_config.heads,
                embed_dim=self.text_config.dim,
                rotary_qk=self.text_config.rotary_qk,
                dtype=jnp.float32,
            )
        else:
            raise NotImplementedError(
                "Only plain transformer encoders are currently supported for the text trunk."
            )

        if self.audio_config.kind == "vit":
            self.audio_encoder = ViT(
                output_dim=self.audio_config.projection_dim,
                num_layers=self.audio_config.depth,
                num_heads=self.audio_config.heads,
                embed_dim=self.audio_config.dim,
                patch_shape=tuple(self.audio_config.patch_shape),
                rotary_qk=self.audio_config.rotary_qk,
            )
        elif self.audio_config.kind == "tnt":
            self.audio_encoder = TNT(
                output_dim=self.audio_config.projection_dim,
                num_layers=self.audio_config.depth,
                inner_num_heads=self.audio_config.inner.heads,
                outer_num_heads=self.audio_config.outer.heads,
                inner_embed_dim=self.audio_config.inner.dim,
                outer_embed_dim=self.audio_config.outer.dim,
                patch_shape=tuple(self.audio_config.outer.patch_shape),
                transformed_patch_shape=tuple(self.audio_config.inner.patch_shape),
                rotary_qk=self.audio_config.rotary_qk,
            )
        elif self.audio_config.kind == "cait":
            self.audio_encoder = CaiT(
                output_dim=self.audio_config.projection_dim,
                num_layers=self.audio_config.depth,
                num_layers_token_only=self.audio_config.token_only_depth,
                num_heads=self.audio_config.heads,
                embed_dim=self.audio_config.dim,
                patch_shape=self.audio_config.patch_shape,
                stoch_depth_rate=self.audio_config.stochastic_depth_rate,
                layerscale_eps=self.audio_config.layerscale_eps,
                rotary_qk=self.audio_config.rotary_qk,
            )
        elif self.audio_config.kind == "mixer":
            self.audio_encoder = MLPMixer(
                output_dim=self.audio_config.projection_dim,
                num_layers=self.audio_config.depth,
                embed_dim=self.audio_config.dim,
                patch_shape=self.audio_config.patch_shape,
            )
        else:
            raise NotImplementedError(
                "Only ViT, TNT, CaiT, and MLPMixer are supported audio trunks."
            )

        self.text_tokenizer = nn.Embed(
            num_embeddings=self.text_config.vocab, features=self.text_config.dim
        )

        self.temp = self.param("temperature", self.temp_init, tuple())

    def encode_text(self, text, is_training):
        enc_text = self.text_encoder(text, is_training=is_training)
        return enc_text

    def encode_audio(self, audio, is_training):
        enc_audio = self.audio_encoder(audio, is_training=is_training)
        return enc_audio

    def __call__(self, text, audio, return_loss=True, is_training=False):
        b = text.shape[0]

        to_text_tokens = self.text_tokenizer

        text = to_text_tokens(text)

        enc_text = self.encode_text(text, is_training)
        enc_audio = self.encode_audio(audio, is_training)

        enc_text = enc_text / jnp.linalg.norm(enc_text, axis=-1, keepdims=True)
        enc_audio = enc_audio / jnp.linalg.norm(enc_audio, axis=-1, keepdims=True)

        sim = einsum("i d, j d -> i j", enc_text, enc_audio) * jnp.exp(self.temp)

        if not return_loss:
            return sim

        labels = jnp.arange(b)
        loss = (
            cross_entropy(sim, labels, axis=0) + cross_entropy(sim, labels, axis=1)
        ) / 2
        return loss
