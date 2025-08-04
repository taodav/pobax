from jax import numpy as jnp
from gymnax.environments import spaces

from pobax.envs.wrappers.gymnax import GymnaxWrapper, Observation


class NamedObservationWrapper(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self._env = env

    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        return Observation(obs=obs), env_state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state, action, params
        )
        return Observation(obs=obs), env_state, reward, done, info

    def observation_space(self, params=None):
        return spaces.Dict(
            {"obs": self._env.observation_space(params)} 
        )

    def action_space(self, params):
        return self._env.action_space(params)
    
    def dummy_observation(self, num_env, params=None):
        obs_space = self._env.observation_space(params)
        return Observation(obs=jnp.zeros((1, num_env,) + obs_space.shape, dtype=obs_space.dtype))


# TODO: Pocman Observation Wrapper
