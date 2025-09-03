from functools import partial
from time import time

from flax import struct
import flax.linen as nn
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

# JAX toy datasets for SkipRNNs
# ------------------------------------------------------------
from jax import random
from typing import Dict, Callable


def _batch(sample_fn, key, batch_size, **kwargs) -> Dict[str, jnp.ndarray]:
    keys = random.split(key, batch_size)
    ex = jax.vmap(lambda k: sample_fn(k, **kwargs), in_axes=0, out_axes=1)(keys)
    # ex is a dict of arrays with leading batch dim via vmap
    return ex

# 1) Rare Pulse Counting (mostly zeros; update only on pulses)
#    x: (B,T,1) binary pulses; y: (B,T,1) cumulative count; mask: all ones
def sample_pulse_count(key, T=200, p=0.08):
    key1, key2 = random.split(key)
    pulses = random.bernoulli(key1, p=p, shape=(T,)).astype(jnp.int32)
    x = pulses[:, None].astype(jnp.float32)
    y = jnp.cumsum(pulses)[:, None].astype(jnp.float32)
    mask = jnp.ones((T,), dtype=bool)
    return {"x": x, "y": y, "mask": mask}

def make_pulse_count_batch(key, batch_size=32, T=200, p=0.08):
    return _batch(sample_pulse_count, key, batch_size, T=T, p=p)

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
    out_size: int

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
        u = jax.random.bernoulli(key, p=u_bar).astype(ins.dtype)  # (B, 1) in {0,1}

        new_hidden_state, embedding = nn.GRUCell(features=self.hidden_size)(state.hidden_state, ins)
        new_hidden_state = u * state.hidden_state + (1 - u) * new_hidden_state

        # The next part is taken from the SkipRNN paper.
        # TODO: do alternative versions of this? maybe we do want our u to be recurrent.
        change_in_u_logit = nn.Dense(features=1)(new_hidden_state)
        change_in_u = jax.nn.sigmoid(change_in_u_logit)
        next_ubar = u * change_in_u + (1 - u) * (u_bar + jnp.minimum(change_in_u_logit, 1 - u_bar))

        logits = nn.Dense(features=self.out_size)(embedding)

        # TODO: change this for general predictions
        out = nn.sigmoid(logits)

        return SkipHiddenState(hidden_state=new_hidden_state, u_bar=next_ubar), (out, logits, u, u_bar)

    @staticmethod
    def initialize_carry(batch_size, hidden_size) -> SkipHiddenState:
        new_hs = nn.GRUCell(features=hidden_size).initialize_carry(
            jax.random.PRNGKey(0), (batch_size, hidden_size)
        ).astype(float)
        new_u = jnp.zeros((batch_size, 1)).astype(float)
        return SkipHiddenState(new_hs, new_u)


def train_step(step_state, _, sampler):
    state, rng = step_state
    sampler_rng, rng = jax.random.split(rng)
    samples = sampler(rng)
    x, y = samples['x'], samples['y']
    init_hstate = SkipRNN.initialize_carry(x.shape[1], d_hidden)

    def _loss(params, init_hstate, rng):
        rng, apply_rng = jax.random.split(rng)
        apply_rngs = jax.random.split(apply_rng, x.shape[0])
        dones = jnp.zeros((x.shape[0], x.shape[1]))
        ins = (x, dones)
        _, (y_preds, y_logits, us, u_bars) = state.apply_fn(params, init_hstate, apply_rngs, ins)
        loss_per_ex = (y - y_logits) ** 2
        # # binary cross entropy
        # loss_per_ex = optax.losses.sigmoid_binary_cross_entropy(y_logits, y)

        logp = jnp.log(u_bars + 1e-10)  # (B,)

        g = jax.lax.stop_gradient(loss_per_ex)
        g_grad_log_pi = (g * (-logp)).mean()

        pathwise_loss = loss_per_ex.mean()

        total_loss = pathwise_loss + g_grad_log_pi

        aux = {
            'y_pred': y_preds,
            'u_bar': u_bars,
            'u': us,
            'total_loss': total_loss,
            'pathwise_loss': pathwise_loss,
            'g_grad_log_pi': g_grad_log_pi,
        }

        return total_loss, aux

    grad_fn = jax.value_and_grad(_loss, has_aux=True)
    grad_rng, rng = jax.random.split(rng)
    total_loss, grads = grad_fn(
        state.params, init_hstate, grad_rng
    )
    state = state.apply_gradients(grads=grads)

    return (state, rng), total_loss


def train(state: TrainState, rng: jax.random.PRNGKey,
          steps: int = int(1e5), log_every: int = 50,
          sampler: Callable = make_pulse_count_batch):
    epochs = steps // log_every
    _train_step = partial(train_step, sampler=sampler)

    def _epoch(epoch_state, i):

        new_epoch_state, loss_and_aux = jax.lax.scan(_train_step, epoch_state, jnp.arange(log_every), log_every)
        _, aux = loss_and_aux
        aux['step'] = log_every * (i + 1)

        def callback(info):
            print(f"Mean statistics over {log_every} steps for step {info['step']:3d}  loss={info['total_loss'].mean():.4f}  "
                  # f"p_mean={aux['p_mean']:.3f}  b_rate={aux['b_rate']:.3f}  "
                  f"u_rate={info['u'].mean():.3f}  "
                  f"pathwise={info['pathwise_loss'].mean():.4f}  reinforce={info['g_grad_log_pi'].mean():.4f}")
        jax.debug.callback(callback, aux)
        return new_epoch_state, aux

    final_state, aux = jax.lax.scan(_epoch, (state, rng), jnp.arange(epochs), epochs)
    return final_state, aux


if __name__ == "__main__":
    # jax.disable_jit(True)
    rng = jax.random.PRNGKey(2025)
    # jax.config.update('jax_platform_name', 'cpu')

    b_size = 32
    d_in = 1
    d_hidden = 64
    t = 32
    lr = 1e-5
    steps = 100000

    p = 0.1
    pulse_sampler = partial(make_pulse_count_batch, T=t, batch_size=b_size, p=p)

    sample_dict = pulse_sampler(jax.random.PRNGKey(0))
    x_sample = sample_dict['x']


    # output one at next step, every n steps
    # one_every = 8
    # x = jnp.zeros((t, b_size, d_in))
    # x = x.at[::one_every, :, 0].set(1)
    #
    # y = jnp.zeros((t, b_size))
    # y = y.at[1::one_every, :].set(1)

    _train = jax.jit(partial(train, steps=steps, log_every=100))

    tx = optax.adam(lr)
    init_x = (
        jnp.zeros((1,) + x_sample.shape[1:]),
        jnp.zeros((1, b_size)),
    )

    network = SkipRNN(hidden_size=d_hidden, out_size=1)
    init_hstate = SkipRNN.initialize_carry(b_size, d_hidden)
    rng, init_rng, call_rng = jax.random.split(rng, 3)
    call_rngs = jax.random.split(call_rng, (1, ))
    network_params = network.init(init_rng, init_hstate, call_rngs, init_x)

    state = TrainState.create(
        apply_fn=network.apply,
        params=network_params,
        tx=tx,
    )

    t = time()

    final_state, final_out = jax.block_until_ready(_train(state, rng))

    new_t = time()
    print()

