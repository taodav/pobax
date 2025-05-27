from functools import partial
from typing import Optional, Tuple, Union

import chex
from flax import struct
import jax
import jax.numpy as jnp
from gymnax.environments import environment, spaces

from pobax.envs.wrappers.gymnax import GymnaxWrapper


@struct.dataclass
class TraceFeatureState:
    env_state: environment.EnvState
    trace_features: jnp.ndarray


class TraceFeaturesWrapper(GymnaxWrapper):
    def __init__(self,
                 env: environment.Environment,
                 lambdas: jnp.ndarray = jnp.array([0., 0.5, 0.7, 0.9, 0.95]),
                 trace_in_obs: bool = False,
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
        self.trace_in_obs = trace_in_obs
        if trace_in_obs:
            assert jnp.any(jnp.isclose(self.lambdas, 0)), 'observation not in trace obs!'

    def observation_space(self, params) -> spaces.Box:
        og_obs_space_shape = self._env.observation_space(params).shape
        if self.trace_in_obs:
            n_lambdas = len(self.lambdas)

            if len(og_obs_space_shape) == 1:
                obs_length = og_obs_space_shape[0]
                shape = (obs_length * n_lambdas,)
            elif len(og_obs_space_shape) == 3:
                # images
                shape = (og_obs_space_shape[0], og_obs_space_shape[1], og_obs_space_shape[2] * n_lambdas)
            else:
                raise NotImplementedError

            return spaces.Box(
                low=self._env.observation_space(params).low,
                high=self._env.observation_space(params).high,
                shape=shape,
                dtype=self._env.observation_space(params).dtype,
            )
        else:
            return self._env.observation_space(params)

    @partial(jax.jit, static_argnums=(0, -1))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, TraceFeatureState]:
        obs, state = self._env.reset(key, params)

        trace_features = obs[..., None].repeat(len(self.lambdas), axis=-1)

        state = TraceFeatureState(env_state=state,
                                  trace_features=trace_features)
        if self.trace_in_obs:
            # here we assume that the last two dimensions
            unflatten_dims = trace_features.shape[:-2]
            obs = trace_features.reshape((*unflatten_dims, -1))
            print('Replacing obs with trace features')

        return obs, state

    @partial(jax.jit, static_argnums=(0,-1))
    def step(
            self,
            key: chex.PRNGKey,
            state: TraceFeatureState,
            action: Union[int, float],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, TraceFeatureState, float, bool, dict]:
        obs, next_state, reward, done, info = self._env.step(key, state.env_state, action, params)

        leading_dims = (1,) * len(obs.shape)
        lambdas = jnp.broadcast_to(self.lambdas, leading_dims + self.lambdas.shape)

        curr_obs_lambda = (1 - done) * (1 - lambdas) + done

        next_trace = (1 - done) * lambdas * state.trace_features + curr_obs_lambda * obs[..., None]

        next_state = TraceFeatureState(env_state=next_state,
                                       trace_features=next_trace)

        if self.trace_in_obs:
            # here we assume that the last two dimensions
            unflatten_dims = next_trace.shape[:-2]
            obs = next_trace.reshape((*unflatten_dims, -1))

        return obs, next_state, reward, done, info