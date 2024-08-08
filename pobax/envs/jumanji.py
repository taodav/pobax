import jax.numpy as jnp
import numpy as np
import gymnasium as gym
# import envpool
from jumanji import specs
from jumanji.specs import Array, BoundedArray, DiscreteArray, MultiDiscreteArray, Spec
from flax import struct
from functools import partial
from typing import Optional, Tuple, Union, Any
from gymnax.environments import environment, spaces

def jumanji_specs_to_gymnax_spaces(spec) -> spaces.Space:
    """Converts Jumanji spec to equivalent Gymnax space."""
    if isinstance(spec, DiscreteArray):
        return spaces.Discrete(spec.num_values)
    elif isinstance(spec, MultiDiscreteArray):
        raise NotImplementedError("MultiDiscrete spaces are not yet supported in Gymnax.")
    elif isinstance(spec, BoundedArray):
        return spaces.Box(low=jnp.array(spec.minimum), high=jnp.array(spec.maximum), shape=spec.shape, dtype=spec.dtype)
    elif isinstance(spec, Array):
        return spaces.Box(low=-jnp.inf, high=jnp.inf, shape=spec.shape, dtype=spec.dtype)
    else:
        # Handling nested specs assuming they are composed of other Spec instances
        return spaces.Dict({
            key: jumanji_specs_to_gymnax_spaces(value)
            for key, value in vars(spec).items()
            if isinstance(value, Spec)
        })

def jumanji_specs_to_gymnasium_spaces(
    spec: Spec,
) -> Union[
    gym.spaces.Box,
    gym.spaces.Discrete,
    gym.spaces.MultiDiscrete,
    gym.spaces.Space,
    gym.spaces.Dict,
]:
    """Converts jumanji specs to gymnasium spaces.

    Args:
        spec: jumanji spec of type jumanji.specs.Spec, can be an Array or any nested spec.

    Returns:
        gymnasium.spaces object corresponding to the equivalent jumanji specs implementation.
    """
    if isinstance(spec, DiscreteArray):
        # Assuming gymnasium has not changed the constructor for Discrete spaces
        return gym.spaces.Discrete(n=spec.num_values)
    elif isinstance(spec, MultiDiscreteArray):
        # Assuming gymnasium has not changed the constructor for MultiDiscrete spaces
        return gym.spaces.MultiDiscrete(nvec=spec.num_values)
    elif isinstance(spec, BoundedArray):
        # Handling potential changes in NumPy and type issues
        low = jnp.broadcast_to(spec.minimum, shape=spec.shape)
        high = jnp.broadcast_to(spec.maximum, shape=spec.shape)
        return gym.spaces.Box(
            low=low,
            high=high,
            shape=spec.shape,
            dtype=spec.dtype
        )
    elif isinstance(spec, Array):
        # Infinite box space handling for unbounded arrays
        return gym.spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=spec.shape,
            dtype=spec.dtype
        )
    else:
        # Handling nested specs assuming they are composed of other Spec instances
        return gym.spaces.Dict(
            {
                key: jumanji_specs_to_gymnasium_spaces(value)
                for key, value in vars(spec).items()
                if isinstance(value, Spec)
            }
        )
    
def jumanji_to_gymnax_obs(observation):
    """Convert a Jumanji observation into a gym observation.

    Args:
        observation: JAX pytree with (possibly nested) containers that
            either have the `__dict__` or `_asdict` methods implemented.

    Returns:
        Numpy array or nested dictionary of numpy arrays.
    """
    if isinstance(observation, jnp.ndarray):
        return observation
    elif hasattr(observation, "__dict__"):
        # Applies to various containers including `chex.dataclass`
        return {
            key: jumanji_to_gymnax_obs(value) for key, value in vars(observation).items()
        }
    elif hasattr(observation, "_asdict"):
        # Applies to `NamedTuple` container.
        return {
            key: jumanji_to_gymnax_obs(value)
            for key, value in observation._asdict().items()  # type: ignore
        }
    else:
        raise NotImplementedError(
            "Conversion only implemented for JAX pytrees with (possibly nested) containers "
            "that either have the `__dict__` or `_asdict` methods implemented."
        )