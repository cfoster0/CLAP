import click
from click_option_group import optgroup

import jax
from jax import random, numpy as np, value_and_grad, jit, tree_util
from optax import (
    chain,
    clip_by_global_norm,
    scale_by_adam,
    scale,
    apply_updates,
    add_decayed_weights,
    masked,
)

from clap.models import CLAP

# data

from torch.utils.data import DataLoader
from clap.datasets import (
    pair_text_spectrogram_dataset_collate_fn,
    PairTextSpectrogramDataset,
)


@click.command()
@optgroup.group("Model settings")
@optgroup.option("--text_vocab", default=256, type=int)
@optgroup.option("--text_dim", default=768, type=int)
@optgroup.option("--text_depth", default=1, type=int)
@optgroup.option("--text_heads", default=8, type=int)
@optgroup.option("--audio_dim", default=512, type=int)
@optgroup.option("--audio_depth", default=1, type=int)
@optgroup.option("--audio_heads", default=8, type=int)
@optgroup.group("Training settings")
@optgroup.option("--data_folder", default="./data", type=str)
@optgroup.option("--batch_size", default=16, type=int)
@optgroup.option("--epochs", default=100, type=int)
@optgroup.option("--learning_rate", default=3e-4, type=float)
@optgroup.option("--weight_decay", default=1e-1, type=float)
@optgroup.option("--seed", default=0, type=int)
@optgroup.option("--max_norm", default=0.5, type=float)
def train(
    *,
    data_folder,
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
    audio_heads,
):
    # rng

    rng_key = random.PRNGKey(seed)

    # data

    dataset = PairTextSpectrogramDataset(data_folder)
    dl = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=pair_text_spectrogram_dataset_collate_fn,
        drop_last=True,
        shuffle=True,
    )

    # model

    model = CLAP(
        text_vocab=text_vocab,
        text_dim=text_dim,
        text_depth=text_depth,
        text_heads=text_heads,
        audio_dim=audio_dim,
        audio_depth=audio_depth,
        audio_heads=audio_heads,
    )

    # optimizer

    exclude_bias = lambda params: tree_util.tree_map(lambda x: x.ndim != 1, params)

    optim = chain(
        clip_by_global_norm(max_norm),
        scale_by_adam(eps=1e-4),
        add_decayed_weights(weight_decay, exclude_bias),
        scale(-learning_rate),
    )

    # init

    audio, audio_mask, text, text_mask = next(iter(dl))

    params = model.init(rng_key, text, audio, text_mask, audio_mask)
    optim_state = optim.init(params)

    # loss function, for use with value_and_grad

    @jit
    @value_and_grad
    def loss_fn(params, text, audio, text_mask, audio_mask):
        return model.apply(params, text, audio, text_mask, audio_mask)

    # train loop

    for _ in range(epochs):
        for audio, audio_mask, text, text_mask in dl:
            loss, grads = loss_fn(
                params, text, audio, text_mask, audio_mask, is_training=True
            )
            updates, optim_state = optim.update(grads, optim_state, params)
            params = apply_updates(params, updates)
            print(f"loss: {loss}")

    # finished


if __name__ == "__main__":
    train()
