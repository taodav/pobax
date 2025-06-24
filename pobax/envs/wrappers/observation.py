from pobax.envs.wrappers.gymnax import GymnaxWrapper
import jax
import numpy as np
from jax import numpy as jnp
from typing import NamedTuple
from gymnax.environments import environment, spaces

class Observation(NamedTuple):
    obs: jnp.ndarray
    action_mask: jnp.ndarray = None

class GeneralObservationWrapper(GymnaxWrapper):
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

class BattleShipObservationWrapper(GeneralObservationWrapper):
    def __init__(self, env, params=None):
        super().__init__(env)
        self._env = env
        self.action_size = env.action_space(params).n
    
    def reset(self, key, params=None):
        obs, env_state = self._env.reset(key, params)
        hit = obs[..., 0:1]
        if len(obs.shape) == 4:
            valid_action_mask = (obs == 0).reshape(*obs.shape[:-2], -1)
        else:
            valid_action_mask = obs[..., 1:self.action_size + 1]
        obs = jnp.concatenate([hit, obs[..., self.action_size + 1:]], axis=-1)
        obs = Observation(obs=obs, action_mask=valid_action_mask)
        return obs, env_state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        hit = obs[..., 0:1]
        if len(obs.shape) == 4:
            valid_action_mask = (obs == 0).reshape(*obs.shape[:-2], -1)
        else:
            valid_action_mask = obs[..., 1:self.action_size + 1]
        obs = jnp.concatenate([hit, obs[..., self.action_size + 1:]], axis=-1)
        obs = Observation(obs=obs, action_mask=valid_action_mask)
        return obs, state, reward, done, info

    def observation_space(self, params=None):
        return spaces.Dict(
            {
                "obs": spaces.Box(
                    low=self._env.observation_space(params).low,
                    high=self._env.observation_space(params).high,
                    shape=self._env.observation_space(params).shape[:-1] + (self._env.observation_space(params).shape[-1] - self.action_size,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.Box(
                    low=0.0,
                    high=1.0,
                    shape=(self.action_size,),
                    dtype=np.float32,
                ),
            }
        )
    
    def dummy_observation(self, num_env, params=None):
        obs_space = self.observation_space(params).spaces["obs"]
        action_mask_shape = (self.action_size,)
        return Observation(
            obs=jnp.zeros((1, num_env,) + obs_space.shape, dtype=obs_space.dtype),
            action_mask=jnp.ones((1, num_env) + action_mask_shape, dtype=jnp.float32)
        )

# TODO: Pocman Observation Wrapper