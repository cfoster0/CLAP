import jax.numpy as jnp
import jax.random as random
import flax.linen as nn


class StochasticDepthBlock(nn.Module):
    drop_rate: float
    scale_by_keep: bool = True

    @nn.compact
    def __call__(self, inputs, is_training: bool):

        if not is_training or self.drop_rate == 0.:
            return inputs

        keep_prob = 1. - self.drop_rate
        rng = self.make_rng('stochastic_depth')

        b = inputs.shape[0]
        shape = [b, ] + ([1, ] * (inputs.ndim - 1))
        random_tensor = random.uniform(rng, shape, dtype=inputs.dtype)
        binary_tensor = jnp.floor(keep_prob + random_tensor)

        if self.scale_by_keep:
            x = inputs / keep_prob

        output = x * binary_tensor
        return output