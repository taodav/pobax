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

class RNNApproximator(ScannedRNN):
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
        ins, _ = x
        rnn_state = self.initialize_carry(ins.shape[0], ins.shape[1])
        
        for _ in range(self.horizon):
            rnn_state, ins = nn.GRUCell(features=self.hidden_size)(rnn_state, ins)
        
        return rnn_state, ins


class SmallImageCNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        # 5x5
        if x.shape[-3] == x.shape[-2] and x.shape[-3] == 5:
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

        elif x.shape[-3] == x.shape[-2] and x.shape[-3] == 64:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(8, 8), strides=2, padding='SAME')(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.hidden_size, kernel_size=(6, 6), strides=2, padding='SAME')(out1)
            out2 = nn.relu(out2)
            out3 = nn.Conv(features=self.hidden_size, kernel_size=(4, 4), strides=2, padding='SAME')(out2)
            out3 = nn.relu(out3)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(3, 3), strides=2, padding='SAME')(out3)

        else:
            raise NotImplementedError

        conv_out = nn.relu(conv_out)
        # Convolutions "flatten" the last three dimensions.
        flat_out = conv_out.reshape((*conv_out.shape[:-3], -1))  # Flatten
        final_out = nn.Dense(features=self.hidden_size)(flat_out)
        return final_out

class SimpleNN(nn.Module):
    hidden_size: int
    depth: int = 3

    @nn.compact
    def __call__(self, x):
        out = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        out = nn.relu(out)
        for _ in range(self.depth - 2):
            out = nn.Dense(self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(out)
            out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        return out

# class SimpleNN(nn.Module):
#     hidden_size: int
#     depth: int = 3

#     @nn.compact
#     def __call__(self, x):
#         out = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
#             x
#         )
#         out = nn.relu(out)
#         out = nn.Dense(
#             self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
#         )(out) 
#         out = nn.relu(out)
#         out = nn.Dense(
#             self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
#         )(out)
#         return out

class ResidualBlock(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        identity = x
        out = nn.Dense(self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        out = nn.relu(out)
        out = nn.Dense(self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(out)
        out += identity
        out = nn.relu(out)
        return out

class SimpleSkipNN(nn.Module):
    hidden_size: int
    depth: int = 3

    @nn.compact
    def __call__(self, x):
        # Apply the first layer with specific initialization
        out = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(x)
        out = nn.relu(out)
        
        for _ in range((self.depth - 1) // 2):
            out = ResidualBlock(self.hidden_size)(out)

        # Apply the last layer without ReLU
        out = nn.Dense(self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(out)
        
        return out
    

class ProbePredictorNN(nn.Module):
    hidden_size: int
    n_outs: int
    n_hidden_layers: int = 1

    @nn.compact
    def __call__(self, x):
        out = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        out = nn.relu(out)

        for i in range(self.n_hidden_layers):
            out = nn.Dense(
                self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
            )(out)
            out = nn.relu(out)

        logits = nn.Dense(
            self.n_outs, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        predictions = nn.sigmoid(logits)
        return predictions, logits
