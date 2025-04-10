from functools import partial
from typing import Optional, Tuple, Union

import chex
from flax import struct
import jax
from jax.numpy import jnp
from gymnax.environments import environment, spaces

from pobax.envs.wrappers.gymnax import GymnaxWrapper


@struct.dataclass
class TraceFeatureState:
    env_state: environment.EnvState
    trace_features: jnp.ndarray


class TraceFeaturesWrapper(GymnaxWrapper):
    def __init__(self,
                 env: environment.Environment,
                 lambdas: jnp.ndarray = jnp.array([0.1, 0.9]),
                 **kwargs):
        """
        TraceFeaturesWrapper adds trace features over observations for each lambda.
        It adds it into a seperate state feature.
        :param env:
        :param lambdas:
        :param kwargs:
        """
        super().__init__(env)
        self.lambdas = lambdas

    @partial(jax.jit, static_argnums=(0,-1))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, TraceFeatureState]:
        obs, state = self._env.reset(key, params)

        state = TraceFeatureState(env_state=state,
                                  trace_features=obs[..., None])

        return obs, state

    @partial(jax.jit, static_argnums=(0,-1))
    def step(
            self,
            key: chex.PRNGKey,
            state: TraceFeatureState,
            action: Union[int, float],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, TraceFeatureState, float, bool, dict]:
        obs, next_state, reward, done, info = self._env.step(key, state, action, params)

        leading_dims = (1,) * len(obs.shape)
        lambdas = jnp.broadcast_to(self.lambdas, leading_dims + self.lambdas.shape)

        next_trace = (1 - done) * lambdas * state.trace_features + obs[..., None]

        next_state = TraceFeatureState(env_state=next_state,
                                       trace_features=next_trace)

        return obs, next_state, reward, done, info
