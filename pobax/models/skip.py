from functools import partial

import jax
from flax import linen as nn
from flax import struct
from jax import numpy as jnp


@struct.dataclass
class SkipHiddenState:
    hidden_state: jnp.ndarray
    u_bar: jnp.ndarray

def round_ste(x: jnp.ndarray) -> jnp.ndarray:
    # forward = round(x); backward d/dx = 1 (passes gradient through)
    return x + jax.lax.stop_gradient(jnp.round(x) - x)


class SkipRNN(nn.Module):
    """
    RNN with a gated skip connection.
    """
    hidden_size: int
    out_size: int
    binarize_type: str = 'round'  # bernoulli | round

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: SkipHiddenState, key: jax.random.PRNGKey, x: jnp.ndarray):
        """Applies the module."""
        state = carry
        ins, resets = x
        reset_hs = self.initialize_carry(ins.shape[0], self.hidden_size)
        state = jax.tree.map(lambda reset_hs, hs: jnp.where(
            resets[:, jnp.newaxis],
            reset_hs,
            hs,
        ), reset_hs, state)

        # binarize in the actor loop. Need to do the u prediction here.
        # Write a small test for this
        u_bar = state.u_bar
        if self.binarize_type == 'bernoulli':
            u = jax.random.bernoulli(key, p=u_bar).astype(ins.dtype)  # (B, 1) in {0,1}
        elif self.binarize_type == 'round':
            u = round_ste(u_bar).astype(ins.dtype)
        else:
            raise NotImplementedError

        new_hidden_state, embedding = nn.GRUCell(features=self.hidden_size)(state.hidden_state, ins)
        new_hidden_state = u * state.hidden_state + (1 - u) * new_hidden_state

        # The next part is taken from the SkipRNN paper.
        # TODO: do alternative versions of this? maybe we do want our u to be recurrent.
        change_in_u_logit = nn.Dense(features=1)(new_hidden_state)
        change_in_u = jax.nn.sigmoid(change_in_u_logit)
        next_ubar = u * change_in_u + (1 - u) * (u_bar + jnp.minimum(change_in_u, 1 - u_bar))

        logits = nn.Dense(features=self.out_size)(embedding)

        # TODO: change this for general predictions
        # out = nn.sigmoid(logits)

        return SkipHiddenState(hidden_state=new_hidden_state, u_bar=next_ubar), (logits, u, u_bar)

    @staticmethod
    def initialize_carry(batch_size, hidden_size) -> SkipHiddenState:
        new_hs = nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        ).astype(float)
        new_u = jnp.zeros((batch_size, 1)).astype(float)
        return SkipHiddenState(new_hs, new_u)
