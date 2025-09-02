# JAX REINFORCE example for a Bernoulli gate (binarizer)
import jax
import jax.numpy as jnp
import optax
from functools import partial
from typing import NamedTuple

key = jax.random.PRNGKey(0)

# ----- simple 2-layer "body" and a separate linear "gate" that outputs a logit -----

def glorot(key, fan_in, fan_out):
    k1, k2 = jax.random.split(key)
    limit = jnp.sqrt(6.0 / (fan_in + fan_out))
    return jax.random.uniform(k1, (fan_in, fan_out), minval=-limit, maxval=limit), jnp.zeros((fan_out,))

def init_params(key, d_in, d_hidden):
    k1, k2, k3 = jax.random.split(key, 3)
    Wg, bg = glorot(k1, d_in, 1)           # gate: logit a = x @ Wg + bg
    W1, b1 = glorot(k2, d_in, d_hidden)    # body
    W2, b2 = glorot(k3, d_hidden, 1)       # head
    return dict(Wg=Wg, bg=bg, W1=W1, b1=b1, W2=W2, b2=b2)

def bernoulli_logprob_from_logits(b, a):
    # a: logits, b in {0,1}
    # log p(b=1) = log_sigmoid(a); log p(b=0) = log_sigmoid(-a)
    return b * jax.nn.log_sigmoid(a) + (1.0 - b) * jax.nn.log_sigmoid(-a)

def forward_sample(params, x, y, key):
    """Sample the gate, run the model, and return per-example loss and log-prob."""
    a = (x @ params["Wg"] + params["bg"]).squeeze(-1)          # logits (B,)
    p = jax.nn.sigmoid(a)
    b = jax.random.bernoulli(key, p).astype(x.dtype)           # (B,) in {0,1}

    h = jax.nn.relu(x @ params["W1"] + params["b1"])           # (B,H)
    h = h * b[:, None]                                         # gate masks the body
    y_hat = (h @ params["W2"] + params["b2"]).squeeze(-1)      # (B,)

    loss_per_ex = (y_hat - y) ** 2                             # MSE per example
    logp = bernoulli_logprob_from_logits(b, a)                 # (B,)
    return loss_per_ex, logp, p, b

# ----- training state -----

class TrainState(NamedTuple):
    params: dict
    opt_state: optax.OptState
    baseline: jnp.ndarray          # scalar EMA baseline
    key: jax.random.PRNGKey

def create_state(key, d_in=16, d_hidden=32, lr=1e-3):
    params = init_params(key, d_in, d_hidden)
    tx = optax.adam(lr)
    return TrainState(params=params,
                      opt_state=tx.init(params),
                      baseline=jnp.array(0.0),
                      key=key)

# ----- loss with REINFORCE term -----

@partial(jax.jit, static_argnames=("beta",))
def train_step(state: TrainState, x, y, beta=0.9):
    """One SGD step; returns new state and metrics."""
    tx = optax.adam(1e-4)

    def total_loss(params, baseline, key):
        k1, k2 = jax.random.split(key)
        loss_per_ex, logp, p, b = forward_sample(params, x, y, k1)

        # Pathwise gradients for differentiable parts
        pathwise = loss_per_ex.mean()

        # Score-function (REINFORCE) term; detach loss to avoid backprop via logp
        adv = jax.lax.stop_gradient(loss_per_ex) - baseline
        reinforce = (adv * (-logp)).mean()  # minimize expected loss

        total = pathwise + reinforce
        aux = dict(
            pathwise=pathwise,
            reinforce=reinforce,
            loss_mean=loss_per_ex.mean(),
            p_mean=p.mean(),
            b_rate=b.mean(),
        )
        return total, aux

    (tot, aux), grads = jax.value_and_grad(total_loss, has_aux=True)(
        state.params, state.baseline, state.key
    )

    updates, new_opt_state = tx.update(grads, state.opt_state, state.params)
    new_params = optax.apply_updates(state.params, updates)

    # EMA baseline update (no gradient)
    new_baseline = beta * state.baseline + (1.0 - beta) * aux["loss_mean"]

    new_key = jax.random.split(state.key, 2)[1]
    new_state = TrainState(new_params, new_opt_state, new_baseline, new_key)
    return new_state, aux



# ----- demo run -----
jax.disable_jit(True)

B, d_in, d_hidden = 64, 16, 32
state = create_state(key, d_in, d_hidden, lr=1e-4)

x = jax.random.normal(state.key, (B, d_in))
y = jax.random.normal(state.key, (B,))

for step in range(10000):
    state, aux = train_step(state, x, y)
    if step % 50 == 0:
        print(f"step {step:3d}  loss={aux['loss_mean']:.4f}  "
              f"p_mean={aux['p_mean']:.3f}  b_rate={aux['b_rate']:.3f}  "
              f"pathwise={aux['pathwise']:.4f}  reinforce={aux['reinforce']:.4f}")

print()
