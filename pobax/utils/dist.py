# stuff needed for distributional RL
import jax
import jax.numpy as jnp


# Normalize input values using a running scale of the range between a given range of percentiles.
def percentile_normalization(x: jax.Array,
                             prev_scale: jax.Array,
                             percentile_range: jax.Array = jnp.array([5, 95]),
                             tau: float = 0.01) -> jax.Array:
    # Compute percentiles for the input values.
    percentiles = jnp.percentile(x, percentile_range)
    scale = percentiles[1] - percentiles[0]

    return tau * scale + (1 - tau) * prev_scale


def sg(x): return jax.tree.map(jax.lax.stop_gradient, x)


def symlog(x: jax.Array) -> jax.Array:
    return jnp.sign(x) * jnp.log(1 + jnp.abs(x))


def symexp(x):
    return jnp.sign(x) * (jnp.exp(jnp.abs(x)) - 1)


def two_hot(x: jax.Array, low: float, high: float, num_bins: int) -> jax.Array:
    """
    Generate two-hot encoded tensor from input tensor.

    Parameters
    ----------
    x : jax.Array
        Input tensor of continuous values. Shape: (*batch_dim, num_values)
        Should **not** have a leading singleton dimension at the end.
    low : float
        Minimum value under consideration in log-space
    high : float
        Maximum value under consideration in log-space
    num_bins : int
        Number of encoding bins

    Returns
    -------
    jax.Array
        _description_
    """
    bin_size = (high - low) / (num_bins - 1)

    x = jnp.clip(symlog(x), low, high)
    bin_index = jnp.floor((x - low) / bin_size).astype(int)
    bin_offset = ((x - low) / bin_size - bin_index.astype(float))

    # Two-hot encode
    two_hot = jax.nn.one_hot(bin_index, num_bins) * (1 - bin_offset[..., None]) + \
              jax.nn.one_hot(bin_index + 1, num_bins) * bin_offset[..., None]

    return two_hot


def two_hot_inv(x: jax.Array,
                low: float, high: float, num_bins: int,
                apply_softmax: bool = True) -> jax.Array:

    bins = jnp.linspace(low, high, num_bins)

    if apply_softmax:
        x = jax.nn.softmax(x, axis=-1)

    x = jnp.sum(x * bins, axis=-1)
    return symexp(x)


def soft_crossentropy(pred_logits: jax.Array, target: jax.Array,
                      low: float, high: float, num_bins: int) -> jax.Array:
    pred = jax.nn.log_softmax(pred_logits, axis=-1)
    target = two_hot(target, low, high, num_bins)
    return -(pred * target).sum(axis=-1)

