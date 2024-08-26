import functools
from typing import Callable, Optional

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax._src.nn.initializers import orthogonal, constant
import numpy as np

from pobax.utils.math import simnorm


class ScannedRNN(nn.Module):
    hidden_size: int

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )
        new_rnn_state, y = nn.GRUCell(features=self.hidden_size)(rnn_state, ins)
        return new_rnn_state, y

    @staticmethod
    def initialize_carry(batch_size, hidden_size):
        # Use a dummy key since the default state init fn is just zeros.
        return nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        )


class FixedHorizonPlanningRNN(ScannedRNN):
    horizon: int = 3

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry, x):
        """Applies the module."""
        rnn_state = carry
        ins, resets = x
        rnn_state = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(ins.shape[0], ins.shape[1]),
            rnn_state,
        )

        def apply_n_times(rnn_state):
            rnn_state, y = nn.GRUCell(features=self.hidden_size)(rnn_state, ins)
            return rnn_state, y

        outs, all_outs = jax.lax.scan(
            apply_n_times, rnn_state, None, self.horizon
        )
        new_rnn_state, y = outs
        return new_rnn_state, y


class SmallImageCNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        num_dims = len(x.shape) - 2
        # 10x10 2 dimensions
        if num_dims == 2 and x.shape[-2] == x.shape[-1] and x.shape[-2] == 10:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=5, strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.hidden_size, kernel_size=4, strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=3, strides=1, padding=0)(out2)

        # 5x5
        elif x.shape[-3] == x.shape[-2] and x.shape[-3] == 5:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(4, 4), strides=1, padding=1)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.hidden_size, kernel_size=(3, 3), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out2)

        # 3x3
        elif x.shape[-3] == x.shape[-2] and x.shape[-3] == 3:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out1)

        # 10x10
        elif x.shape[-3] == x.shape[-2] and x.shape[-3] == 10:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(5, 5), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.hidden_size, kernel_size=(4, 4), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(3, 3), strides=1, padding=0)(out2)

        else:
            raise NotImplementedError

        conv_out = nn.relu(conv_out)
        # Convolutions "flatten" the last num_dims dimensions.
        flat_out = conv_out.reshape((*conv_out.shape[:-num_dims], -1))  # Flatten
        final_out = nn.Dense(features=self.hidden_size)(flat_out)
        return final_out


class SimpleNN(nn.Module):
    hidden_size: int

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
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        return out


class NormedLinear(nn.Module):
  features: int
  activation: Callable[[jax.Array], jax.Array] = None
  dropout_rate: Optional[float] = None
  norm: nn.Module = nn.LayerNorm

  kernel_init: Callable = nn.initializers.truncated_normal(stddev=0.02)
  dtype: jnp.dtype = jnp.bfloat16  # Switch this to bfloat16 for speed
  param_dtype: jnp.dtype = jnp.float32

  @nn.compact
  def __call__(self, x: jax.Array, training: bool = True) -> jax.Array:
    x = nn.Dense(features=self.features,
                 kernel_init=self.kernel_init,
                 bias_init=nn.initializers.zeros_init(),
                 dtype=self.dtype,
                 param_dtype=self.param_dtype)(x)

    x = self.norm(dtype=self.dtype)(x)

    if self.activation is not None:
      x = self.activation(x)

    if self.dropout_rate is not None and self.dropout_rate > 0:
      x = nn.Dropout(rate=self.dropout_rate)(x, deterministic=not training)

    return x


class Ensemble(nn.Module):
    base_module: nn.Module
    num: int = 2

    @nn.compact
    def __call__(self, *args, **kwargs):
        ensemble = nn.vmap(self.base_module,
                           variable_axes={'params': 0},
                           split_rngs={
                               'params': True,
                               'dropout': True
                           },
                           in_axes=None,
                           out_axes=0,
                           axis_size=self.num)
        return ensemble()(*args, **kwargs)


class TDMPC2ImageCNN(nn.Module):
    simnorm_dim: int = 8

    @nn.compact
    def __call__(self, x):
        num_features = x.shape[-3]

        # preprocessing
        # TODO(?): random augmentations to images
        # https://github.com/nicklashansen/tdmpc2/blob/main/tdmpc2/common/layers.py#L27

        # normalizing pixels
        x = x / 255 - 0.5

        # 64x64 2 dimensions
        if x.shape[-2] == x.shape[-1] and x.shape[-2] == 64:
            out1 = nn.Conv(features=num_features, kernel_size=7, strides=2, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=num_features, kernel_size=5, strides=2, padding=0)(out1)
            out2 = nn.relu(out2)
            out3 = nn.Conv(features=num_features, kernel_size=3, strides=2, padding=0)(out2)
            out3 = nn.relu(out3)
            conv_out = nn.Conv(features=num_features, kernel_size=3, strides=1, padding=0)(out3)
        elif x.shape[-2] == x.shape[-1] and x.shape[-2] == 32:
            out1 = nn.Conv(features=num_features, kernel_size=4, strides=2, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=num_features, kernel_size=3, strides=2, padding=0)(out1)
            out2 = nn.relu(out2)
            out3 = nn.Conv(features=num_features, kernel_size=3, strides=1, padding=0)(out2)
            out3 = nn.relu(out3)
            conv_out = nn.Conv(features=num_features, kernel_size=2, strides=1, padding=0)(out3)

        # Convolutions "flatten" the last num_dims dimensions.
        flat_out = conv_out.reshape((*conv_out.shape[:-3], -1))  # Flatten
        final_out = simnorm(flat_out, simplex_dim=self.simnorm_dim)
        return final_out
