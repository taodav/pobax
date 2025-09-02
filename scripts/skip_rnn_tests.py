from functools import partial

from flax import struct
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

@struct.dataclass
class SkipHiddenState:
    hidden_state: jnp.ndarray
    u_bar: jnp.ndarray

def bernoulli_logprob_from_logits(b, a):
    # a: logits, b in {0,1}
    # log p(b=1) = log_sigmoid(a); log p(b=0) = log_sigmoid(-a)
    return b * jax.nn.log_sigmoid(a) + (1.0 - b) * jax.nn.log_sigmoid(-a)

class SkipRNN(nn.Module):
    """
    RNN with a gated skip connection.
    """
    hidden_size: int

    @partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0,
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self, carry: SkipHiddenState, key, x):
        """Applies the module."""
        state = carry
        ins, resets = x
        state = jax.tree.map(lambda reset_hs, hs: jnp.where(
            resets[:, jnp.newaxis],
            reset_hs,
            hs,
        ), self.initialize_carry(ins.shape[0], ins.shape[1]), state)

        # binarize in the actor loop. Need to do the u prediction here.
        # Write a small test for this
        u_bar = state.u_bar
        u = jax.random.bernoulli(key, u_bar).astype(x.dtype)  # (B,) in {0,1}

        new_hidden_state, y = nn.GRUCell(features=self.hidden_size)(state.hidden_state, ins)
        new_hidden_state = u[:, None] * state.hidden_state + (1 - u[:, None]) * new_hidden_state

        # The next part is taken from the SkipRNN paper.
        # TODO: do alternative versions of this? maybe we do want our u to be recurrent.
        change_in_u_logit = nn.Dense(features=new_hidden_state.shape[-1])(new_hidden_state)
        change_in_u = jax.nn.sigmoid(change_in_u_logit)
        next_ubar = u * change_in_u + (1 - u) * (u_bar + jnp.minimum(change_in_u_logit, 1 - u_bar))

        return SkipHiddenState(hidden_state=new_hidden_state, u_bar=next_ubar), y, u

    @staticmethod
    def initialize_carry(batch_size, hidden_size) -> SkipHiddenState:
        new_hs = nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        ).astype(float)
        new_u = jnp.zeros(batch_size).astype(float)
        return SkipHiddenState(new_hs, new_u)

def train_step(state: TrainState, x: jnp.ndarray, y: jnp.ndarray, lr: float = 1e-4):
    tx = optax.adam(lr)

if __name__ == "__main__":
    b_size = 32
    d_in = 1
    d_hidden = 32
    t = 128 + 1

    one_every = 8

    x = jnp.zeros((b_size, t, d_in))
    x = x.at[:, ::one_every, 0].set(1)

    y = jnp.zeros((b_size, t))
    y = y.at[:, 1::one_every].set(1)

    _train_step = jax.jit(partial(train_step, lr=1e-4))

    init_x = (
        x[:1],
        jnp.zeros((1, 1)),
    )

    for step in range(10000):
        state, aux = train_step(state, x, y)
        if step % 50 == 0:
            print(f"step {step:3d}  loss={aux['loss_mean']:.4f}  "
                  f"p_mean={aux['p_mean']:.3f}  b_rate={aux['b_rate']:.3f}  "
                  f"pathwise={aux['pathwise']:.4f}  reinforce={aux['reinforce']:.4f}")

