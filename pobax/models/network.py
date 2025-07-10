import functools

import flax.linen as nn
import jax
import jax.numpy as jnp
from jax._src.nn.initializers import orthogonal, constant
import numpy as np


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
        if len(x.shape) == 4:
            num_dims = 3
        else:
            num_dims = len(x.shape) - 2  # b x num_envs
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

        elif x.shape[-2] == 7 and x.shape[-3] == 4:
            out1 = nn.Conv(features=64, kernel_size=(2, 4), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=128, kernel_size=(2, 3), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out2)
        elif x.shape[-2] == 5 and x.shape[-3] == 3:
            out1 = nn.Conv(features=64, kernel_size=(2, 3), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            conv_out = nn.Conv(features=128, kernel_size=(2, 2), strides=1, padding=0)(out1)
            # out2 = nn.relu(out2)
            # conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out2)

        elif x.shape[-2] == 3 and x.shape[-3] == 2:
            out1 = nn.Conv(features=64, kernel_size=(1, 1), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            conv_out = nn.Conv(features=128, kernel_size=(2, 2), strides=1, padding=0)(out1)

        elif x.shape[-2] >= 14:
            out1 = nn.Conv(features=64, kernel_size=(6, 6), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=64, kernel_size=(5, 5), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)

            final_out = out2
            # if x.shape[-2] >= 20:
            #     out3 = nn.Conv(features=64, kernel_size=(3, 3), strides=1, padding=0)(out2)
            #     out3 = nn.relu(out3)
            #     final_out = out3
            conv_out = nn.Conv(features=64, kernel_size=(2, 2), strides=1, padding=0)(final_out)

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
        out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        return out


class FullImageCNN(nn.Module):
    hidden_size: int
    num_channels: int = 32

    @nn.compact
    def __call__(self, x):
        if len(x.shape) == 4:
            num_dims = 3
        else:
            num_dims = len(x.shape) - 2  # b x num_envs
        out1 = nn.Conv(features=self.num_channels, kernel_size=(7, 7), strides=4)(x)
        out1 = nn.relu(out1)
        out2 = nn.Conv(features=self.num_channels, kernel_size=(5, 5), strides=2)(out1)
        out2 = nn.relu(out2)
        out3 = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=2)(out2)
        out3 = nn.relu(out3)
        out4 = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=2)(out3)
        flat_out = out4.reshape((*out4.shape[:-num_dims], -1))  # Flatten
        flat_out = nn.relu(flat_out)

        dense_out = nn.Dense(features=self.hidden_size)(flat_out)
        dense_out = nn.relu(dense_out)

        final_out = nn.Dense(features=self.hidden_size)(dense_out)
        return final_out