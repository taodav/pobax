from typing import Union

import distrax
import flax.linen as nn
import jax.numpy as jnp
from jax._src.nn.initializers import orthogonal, constant

from gymnax.environments import spaces


class Actor(nn.Module):
    action_space: Union[spaces.Discrete, spaces.Box]
    hidden_size: int = 128
    # activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if isinstance(self.action_space, spaces.Discrete):
            actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                x
            )
            actor_mean = nn.relu(actor_mean)
            actor_mean = nn.Dense(
                self.action_space.n, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
            pi = distrax.Categorical(logits=actor_mean)
        elif isinstance(self.action_space, spaces.Box):
            actor_mean = nn.Dense(
                2 * self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
            )(x)
            actor_mean = nn.tanh(actor_mean)
            actor_mean = nn.Dense(
                self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(actor_mean)
            actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
            pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        else:
            raise NotImplementedError

        return pi
