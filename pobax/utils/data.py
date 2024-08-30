from collections import namedtuple
import functools
from typing import Callable, Optional

import numpy as np
import jax
from jax import numpy as jnp

ACParams = namedtuple('ACParams', ['actor', 'critic'])
SACParams = namedtuple('SACParams', ['actor', 'critic', 'temperature'])

def add_dim_to_args(
        func: Callable,
        axis: int = 1,
        starting_arg_index: Optional[int] = 1,
        ending_arg_index: Optional[int] = None,
        kwargs_on_device_keys: Optional[list] = None,
):
    """Adds a dimension to the specified arguments of a function.

    Args:
        func (Callable): The function to wrap.
        axis (int, optional): The axis to add the dimension to. Defaults to 1.
        starting_arg_index (Optional[int], optional): The index of the first argument to
            add the dimension to. Defaults to 1.
        ending_arg_index (Optional[int], optional): The index of the last argument to
            add the dimension to. Defaults to None.
        kwargs_on_device_keys (Optional[list], optional): The keys of the kwargs that should
            be added to. Defaults to None.
    """

    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if ending_arg_index is None:
            end_index = len(args)
        else:
            end_index = ending_arg_index

        args = list(args)
        args[starting_arg_index:end_index] = [
            jax.tree.map(lambda x: jnp.expand_dims(x, axis=axis), a)
            for a in args[starting_arg_index:end_index]
        ]
        for k, v in kwargs.items():
            if kwargs_on_device_keys is None or k in kwargs_on_device_keys:
                kwargs[k] = jax.tree.map(lambda x: jnp.expand_dims(x, axis=1), v)
        return func(*args, **kwargs)

    return wrapper


def one_hot(x, n):
    return np.eye(n)[x]