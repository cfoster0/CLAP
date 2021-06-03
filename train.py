from omegaconf import DictConfig, OmegaConf
import hydra

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

from clap.datasets import PairTextSpectrogramTFRecords


@hydra.main(config_path="configs")
def train(cfg: DictConfig) -> None:

    print(OmegaConf.to_yaml(cfg))

    # rng

    rng_key = random.PRNGKey(cfg.training.seed)

    # data

    training_data_path = hydra.utils.get_original_cwd() + "/" + cfg.training.data_folder
    dataloader = PairTextSpectrogramTFRecords(
        training_data_path,
        cfg.training.batch_size,
        )

    # model

    model = CLAP(
        text_config=cfg.model.text,
        audio_config=cfg.model.audio,
    )

    # optimizer

    exclude_bias = lambda params: tree_util.tree_map(lambda x: x.ndim != 1, params)

    optim = chain(
        clip_by_global_norm(cfg.optimizer.max_norm),
        scale_by_adam(eps=1e-4),
        add_decayed_weights(cfg.optimizer.weight_decay, exclude_bias),
        scale(-cfg.optimizer.learning_rate),
    )

    # init

    batch = next(iter(dataloader))

    text = batch['text']
    audio = batch['audio']

    params = model.init(rng_key, text, audio)
    optim_state = optim.init(params)

    # loss function, for use with value_and_grad

    @jit
    @value_and_grad
    def loss_fn(params, text, audio):
        return model.apply(
            params,
            text,
            audio,
            return_loss=True,
            is_training=True,
        )

    # train loop

    for _ in range(cfg.training.epochs):
        for batch in dataloader:
            text = batch['text']
            audio = batch['audio']
            loss, grads = loss_fn(params, text, audio)
            updates, optim_state = optim.update(grads, optim_state, params)
            params = apply_updates(params, updates)
            print(f"loss: {loss}")

    # finished


if __name__ == "__main__":
    train()
