import click
from click_option_group import optgroup

import jax
from jax import random, numpy as np, value_and_grad, jit, tree_util
from optax import chain, clip_by_global_norm, scale_by_adam, scale, apply_updates, add_decayed_weights, masked

from clap.models import CLAP

def data(num_samples, batch_size, text_vocab, audio_dim, rng_key):
    for _ in range(num_samples):
        text = random.randint(rng_key, (batch_size, 16,), 0, text_vocab)
        audio = random.uniform(rng_key, (batch_size, 8, audio_dim))

        text_mask = np.ones((batch_size, 16), dtype = bool)
        audio_mask = np.ones((batch_size, 8), dtype = bool)
        yield text, audio, text_mask, audio_mask

@click.command()
@optgroup.group('Model settings')
@optgroup.option('--text_vocab', default = 256, type = int)
@optgroup.option('--text_dim', default = 512, type = int)
@optgroup.option('--text_depth', default = 1, type = int)
@optgroup.option('--text_heads', default = 8, type = int)
@optgroup.option('--audio_dim', default = 512, type = int)
@optgroup.option('--audio_depth', default = 1, type = int)
@optgroup.option('--audio_heads', default = 8, type = int)
@optgroup.group('Training settings')
@optgroup.option('--batch_size', default = 16, type = int)
@optgroup.option('--epochs', default = 20, type = int)
@optgroup.option('--learning_rate', default = 3e-4, type = float)
@optgroup.option('--weight_decay', default = 1e-1, type = float)
@optgroup.option('--seed', default = 0, type = int)
@optgroup.option('--max_norm', default = 0.5, type = float)
def train(
    *,
    batch_size,
    epochs,
    learning_rate,
    weight_decay,
    seed,
    max_norm,
    text_vocab,
    text_dim,
    text_depth,
    text_heads,
    audio_dim,
    audio_depth,
    audio_heads
):
    # rng

    rng_key = random.PRNGKey(seed)

    # model 

    model = CLAP(
        text_vocab = text_vocab,
        text_dim = text_dim,
        text_depth = text_depth,
        text_heads = text_heads,
        audio_dim = audio_dim,
        audio_depth = audio_depth,
        audio_heads = audio_heads
    )

    # optimizer

    exclude_bias = lambda params: tree_util.tree_map(lambda x: x.ndim != 1, params)

    optim = chain(
        clip_by_global_norm(max_norm),
        scale_by_adam(eps=1e-4),
        add_decayed_weights(weight_decay, exclude_bias),
        scale(-learning_rate)
    )

    text, audio, text_mask, audio_mask = next(data(1, batch_size, text_vocab, audio_dim, rng_key))
    params = model.init(rng_key, text, audio, text_mask, audio_mask)
    optim_state = optim.init(params)

    # loss function, for use with value_and_grad

    @jit
    @value_and_grad
    def loss_fn(params, text, audio, text_mask, audio_mask):
        return model.apply(params, text, audio, text_mask, audio_mask)

    # train loop

    for _ in range(epochs):
        for text, audio, text_mask, audio_mask in data(100, batch_size, text_vocab, audio_dim, rng_key):
            loss, grads = loss_fn(params, text, audio, text_mask, audio_mask)
            updates, optim_state = optim.update(grads, optim_state, params)
            params = apply_updates(params, updates)
            print(f'loss: {loss}')

    # finished

if __name__ == "__main__":
    train()
