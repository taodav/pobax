from einops import rearrange

from jax.scipy.ndimage import map_coordinates
from jax import vmap

import jax.numpy as jnp
import jax



def mish(x: jax.Array) -> jax.Array:
    return x * jnp.tanh(jnp.log(1 + jnp.exp(x)))


def simnorm(x: jax.Array, simplex_dim: int = 8) -> jax.Array:
    x = rearrange(x, '...(L V) -> ... L V', V=simplex_dim)
    x = jax.nn.softmax(x, axis=-1)
    return rearrange(x, '... L V -> ... (L V)')


def grid_sample(input, grid):
    assert isinstance(input, jax.Array)
    assert isinstance(grid, jax.Array)
    assert len(input.shape) == 4
    assert len(grid.shape) == 4
    assert input.shape[0] == grid.shape[0]
    assert grid.shape[-1] == 2
    B, C, Hi, Wi = input.shape
    _, Ho, Wo, _ = grid.shape

    coordinates = (
        (jnp.flip(grid, axis=-1) + 1.0) / 2.0 * jnp.array([Hi - 1.0, Wi - 1.0]).reshape(1, 1, 1, 2)
    )
    bilinear_sample_grey = lambda grey, coords: map_coordinates(
        grey, coords.reshape(-1, 2).transpose(), order=1
    )
    bilinear_sample_image = vmap(bilinear_sample_grey, in_axes=[0, None])
    return vmap(bilinear_sample_image)(input, coordinates).reshape(B, C, Ho, Wo)
