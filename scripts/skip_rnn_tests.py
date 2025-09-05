from functools import partial
from time import time

from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import optax

# JAX toy datasets for SkipRNNs
# ------------------------------------------------------------
from jax import random
from typing import Dict, Callable

from pobax.models.skip import SkipRNN


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


def sample_start_zero_pulse_count(key, T=200, p=0.08, portion_start_zeros: float = 0.4):
    n_zeros = int(portion_start_zeros * T)
    n_nonzeros = T - n_zeros
    key1, key2 = random.split(key)

    pulses = random.bernoulli(key1, p=p, shape=(n_nonzeros,)).astype(jnp.int32)
    start_indicators_nonzero = jnp.zeros_like(pulses)
    start_indicators_zero = jnp.zeros(n_zeros)
    start_indicators = jnp.concatenate([start_indicators_zero, start_indicators_nonzero], axis=0)
    pulses = jnp.concatenate([start_indicators_zero, pulses], axis=0)
    x = jnp.stack([pulses, start_indicators], axis=-1).astype(float)
    y = jnp.cumsum(pulses)[:, None].astype(jnp.float32)
    mask = jnp.ones((T,), dtype=bool)
    return {"x": x, "y": y, "mask": mask}


def make_start_zero_pulse_count_batch(key, batch_size=32, T=200, p=0.08):
    return _batch(sample_start_zero_pulse_count, key, batch_size, T=T, p=p)


def bernoulli_logprob_from_logits(b, a):
    # a: logits, b in {0,1}
    # log p(b=1) = log_sigmoid(a); log p(b=0) = log_sigmoid(-a)
    return b * jax.nn.log_sigmoid(a) + (1.0 - b) * jax.nn.log_sigmoid(-a)


def train_step(step_state, _, sampler, binarize_type: str = 'round'):
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
        _, (y_logits, us, u_bars) = state.apply_fn(params, init_hstate, apply_rngs, ins)
        loss_per_ex = (y - y_logits) ** 2
        # # binary cross entropy
        # loss_per_ex = optax.losses.sigmoid_binary_cross_entropy(y_logits, y)
        pathwise_loss = loss_per_ex.mean()

        if binarize_type == 'bernoulli':
            # REINFORCE estimator
            logp = jnp.log(u_bars + 1e-10)  # (B,)

            g = jax.lax.stop_gradient(loss_per_ex)
            g_grad_log_pi = (g * (-logp)).mean()

            total_loss = pathwise_loss + g_grad_log_pi
        elif binarize_type == 'round':
            total_loss = pathwise_loss
            g_grad_log_pi = 0.
        else:
            raise NotImplementedError

        aux = {
            'u_bar': u_bars,
            'u': us,
            'total_loss': total_loss,
            'pathwise_loss': pathwise_loss,
            'g_grad_log_pi': g_grad_log_pi,
            'y_logits': y_logits
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
          sampler: Callable = make_pulse_count_batch,
          binarize_type: str = 'round'):
    epochs = steps // log_every
    _train_step = partial(train_step, sampler=sampler, binarize_type=binarize_type)

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
    rng = jax.random.PRNGKey(2026)
    # jax.config.update('jax_platform_name', 'cpu')

    b_size = 64
    d_hidden = 128
    t = 64
    lr = 1e-5
    steps = 100000
    binarize_type = 'round'

    p = 0.2
    sampler = partial(make_start_zero_pulse_count_batch, T=t, batch_size=b_size, p=p)

    sample_dict = sampler(jax.random.PRNGKey(0))
    x_sample = sample_dict['x']

    _train = jax.jit(partial(train, sampler=sampler, steps=steps, log_every=50, binarize_type=binarize_type))

    tx = optax.adam(lr)
    init_x = (
        jnp.zeros((1,) + x_sample.shape[1:]),
        jnp.zeros((1, b_size)),
    )

    network = SkipRNN(hidden_size=d_hidden, out_size=1, binarize_type=binarize_type)
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

