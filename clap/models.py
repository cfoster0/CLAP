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


def fixed_pos_embedding(seq, dim):
    inv_freq = 1.0 / (10000 ** (jnp.arange(0, dim, 2) / dim))

    sinusoid_inp = jnp.einsum("i , j -> i j", jnp.arange(seq), inv_freq)

    return jnp.sin(sinusoid_inp), jnp.cos(sinusoid_inp)


def rotate_every_two(x):
    x1 = x[:, :, ::2]
    x2 = x[:, :, 1::2]

    x = jnp.stack((-x2, x1), axis=-1)

    return rearrange(x, "... d j -> ... (d j)")


def apply_rotary_pos_emb(x, sincos):
    sin, cos = map(lambda t: repeat(t, "b n -> b (n j)", j=2)[:, None, :], sincos)
    return (x * cos) + (rotate_every_two(x) * sin)


# main class


class Attention(nn.Module):
    dim: int
    heads: int
    dim_head: int = 64
    causal: bool = False

    @nn.compact
    def __call__(self, x, pos_emb, mask):
        dim_in, h = x.shape[-1], self.heads
        scale = dim_in ** -0.5

        norm = nn.LayerNorm()
        to_qkv = nn.Dense(features=self.dim_head * h * 3, use_bias=False)
        to_out = nn.Dense(features=dim_in)

        x = norm(x)
        qkv = jnp.split(to_qkv(x), 3, axis=-1)
        q, k, v = map(lambda t: rearrange(t, "i (h d) -> i h d", h=h), qkv)

        q = index_update(q, index[1:], apply_rotary_pos_emb(q[1:], pos_emb))
        k = index_update(k, index[1:], apply_rotary_pos_emb(k[1:], pos_emb))

        sim = einsum("i h d, j h d -> i j h", q, k) * scale

        mask = jnp.pad(mask, (1, 0), constant_values=True)
        mask = rearrange(mask, "j -> () j ()")

        if self.causal:
            i, j = sim.shape[:2]
            tri_mask = jnp.ones((i - 1, j - 1), dtype=bool)
            tri_mask = jnp.pad(tri_mask, ((1, 0), (1, 0)), constant_values=False)
            causal_mask = jnp.triu(tri_mask, j - i + 1)
            causal_mask = rearrange(causal_mask, "i j -> i j ()")
            mask = ~causal_mask * mask

        sim = jnp.where(mask, sim, LARGE_NEG_VALUE)

        attn = nn.softmax(sim, axis=-2)

        out = einsum("i j h, j h d -> i h d", attn, v)

        out = rearrange(out, "i h d -> i (h d)")
        return to_out(out)


class FeedForward(nn.Module):
    mult: int = 4

    @nn.compact
    def __call__(self, x):
        dim_in, mult = x.shape[-1], self.mult

        norm = nn.LayerNorm()
        to_intermediate = nn.Dense(features=dim_in * mult)
        to_out = nn.Dense(features=dim_in)

        x = norm(x)
        x = to_intermediate(x)
        x = nn.gelu(x)
        x = to_out(x)
        return x


class Transformer(nn.Module):
    dim: int
    depth: int
    heads: int
    dim_head: int = 64
    causal: bool = False

    cls_init: Callable = nn.initializers.lecun_normal()

    def setup(self):
        self.layers = [
            (
                Attention(
                    dim=self.dim,
                    heads=self.heads,
                    dim_head=self.dim_head,
                    causal=self.causal,
                ),
                FeedForward(),
            )
            for _ in range(self.depth)
        ]

    @nn.compact
    def __call__(self, x, mask):
        n, d, h, dh, dim = *x.shape, self.heads, self.dim_head, self.dim

        if d != dim:
            x = nn.Dense(features=dim)(x)

        cls_token = self.param("cls", self.cls_init, (1, x.shape[-1]))
        to_norm_out = nn.LayerNorm()

        sincos = fixed_pos_embedding(n, self.dim_head)

        x = jnp.concatenate((cls_token, x), axis=0)

        for attn, ff in self.layers:
            x = attn(x, pos_emb=sincos, mask=mask) + x
            x = ff(x) + x

        x = to_norm_out(x)
        x = x[0, :]
        return x


class CLAP(nn.Module):
    text_vocab: int
    text_dim: int
    text_depth: int
    text_heads: int

    audio_dim: int
    audio_depth: int
    audio_heads: int

    temp_init: Callable = nn.initializers.zeros

    def setup(self):
        self.audio_encoder = Transformer(
            dim=self.audio_dim, depth=self.audio_depth, heads=self.audio_heads
        )
        #self.text_encoder = Transformer(
        #    dim=self.text_dim, depth=self.text_depth, heads=self.text_heads, causal=True
        #)
        self.text_encoder = create_trunk('txt_b', output_dim=self.text_dim)

        self.text_tokenizer = nn.Embed(num_embeddings=self.text_vocab, features=self.text_dim)

        self.temp = self.param("temperature", self.temp_init, tuple())


    def encode_text(self, text, mask, is_training):
        # Ignore mask, for now.
        enc_text = self.text_encoder(text, is_training=is_training)
        return enc_text
    
    def encode_audio(self, audio, mask, is_training):
        enc_audio = vmap(self.audio_encoder)(audio, mask)
        return enc_audio

    def __call__(self, text, audio, text_mask, audio_mask, return_loss=True, is_training=False):
        b = text.shape[0]

        to_text_tokens = self.text_tokenizer

        text = to_text_tokens(text)

        enc_text = self.encode_text(text, mask=text_mask, is_training=is_training)
        enc_audio = self.encode_audio(audio, mask=audio_mask, is_training=is_training)

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
