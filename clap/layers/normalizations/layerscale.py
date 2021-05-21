import flax.linen as nn
import jax.numpy as jnp


def full(eps: float, dtype: jnp.dtype = jnp.float32):

    def init(key, shape, dtype=dtype):
        return jnp.full(shape, eps, dtype=dtype)

    return init


class LayerScaleBlock(nn.Module):
    eps: float
    dtype: jnp.dtype = jnp.float32

    @nn.compact
    def __call__(self, inputs, *unused_args, **unused_kwargs):
        in_ch = inputs.shape[-1]
        scale = self.param('layerscale', full(self.eps, dtype=self.dtype),
                           (in_ch,))
        scale = jnp.asarray(scale, self.dtype)
        return inputs * scale
