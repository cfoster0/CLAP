import jax
from typing import Any, Callable, Sequence, Optional
from jax import lax, random, numpy as jnp, vmap, jit
from jax.ops import index, index_update
from .trunks import create_trunk

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
    text_vocab: int
    text_dim: int
    text_depth: int
    text_heads: int

    audio_dim: int
    audio_depth: int
    audio_heads: int

    projection_dim: int

    temp_init: Callable = nn.initializers.zeros

    def setup(self):
        self.audio_encoder = create_trunk('spec_b', output_dim=self.projection_dim)

        self.text_encoder = create_trunk('txt_b', output_dim=self.projection_dim)

        self.text_tokenizer = nn.Embed(
            num_embeddings=self.text_vocab, features=self.text_dim
        )

        self.temp = self.param("temperature", self.temp_init, tuple())

    def encode_text(self, text, mask, is_training):
        # Ignore mask, for now.
        enc_text = self.text_encoder(text, is_training=is_training)
        return enc_text

    def encode_audio(self, audio, mask, is_training):
        # Ignore mask, for now.
        enc_audio = self.audio_encoder(audio, is_training=is_training)
        return enc_audio

    def __call__(
        self, text, audio, text_mask, audio_mask, return_loss=True, is_training=False
    ):
        b = text.shape[0]

        to_text_tokens = self.text_tokenizer

        text = to_text_tokens(text)

        enc_text = self.encode_text(text, text_mask, is_training)
        enc_audio = self.encode_audio(audio, audio_mask, is_training)

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
