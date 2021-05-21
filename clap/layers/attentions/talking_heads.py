from flax import linen as nn
from jax import numpy as jnp


class TalkingHeadsBlock(nn.Module):
    num_heads: int

    @nn.compact
    def __call__(self, inputs):
        transform_shape = (self.num_heads, self.num_heads)
        transform = self.param('talking_heads_transform',
                               nn.initializers.orthogonal(), transform_shape)
        output = jnp.einsum('h i, b h ... -> b i ...', transform, inputs)
        return output
