import jax.numpy as jnp
from jax._src.nn.initializers import orthogonal, constant
import flax.linen as nn


class QuantileQ(nn.Module):
    n_atoms: int
    hidden_size: int
    n_actions: int

    @nn.compact
    def __call__(self, x):
        out = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        out = nn.relu(out)
        out = nn.Dense(
            self.n_actions * self.n_atoms, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        quantiles = out.reshape((*x.shape[:-1], self.n_actions, self.n_atoms))
        return quantiles


class QuantileV(nn.Module):
    n_atoms: int
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        v_vals = QuantileQ(self.n_atoms, self.hidden_size, 1)(x)
        return jnp.squeeze(v_vals, axis=-2)
