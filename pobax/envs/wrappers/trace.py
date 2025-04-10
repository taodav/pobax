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
                 lambdas: jnp.ndarray = jnp.array([0.1, 0.9, 0.95]),
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

    def action_size(self, params):
        action_space = self.action_space(params)
        if isinstance(action_space, spaces.Box):
            assert len(action_space.shape) == 1
            action_size = action_space.shape[0]
        elif isinstance(action_space, spaces.Discrete):
            action_size = action_space.n
        else:
            raise NotImplementedError
        return action_size

    @partial(jax.jit, static_argnums=(0,-1))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, TraceFeatureState]:
        obs, state = self._env.reset(key, params)
        action_vec = jnp.zeros(self.action_size(params))

        obs_action = jnp.concatenate((obs, action_vec))

        state = TraceFeatureState(env_state=state,
                                  trace_features=obs_action[..., None])

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

        action_vec = action
        action_space = self.action_space(params)
        if isinstance(action_space, spaces.Discrete):
            action_vec = jnp.eye(action_space.n)[action]

        obs_action = jnp.concatenate((obs, action_vec))

        leading_dims = (1,) * len(obs.shape)
        lambdas = jnp.broadcast_to(self.lambdas, leading_dims + self.lambdas.shape)

        next_trace = (1 - done) * lambdas * state.trace_features + obs_action[..., None]

        next_state = TraceFeatureState(env_state=next_state,
                                       trace_features=next_trace)

        return obs, next_state, reward, done, info
