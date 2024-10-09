from typing import Optional, Tuple, Union

import chex
import jax
import mujoco
from brax import base
from brax.io import image
from gymnax.environments import spaces, environment
from jax import numpy as jnp

from pobax.envs import VecEnv
from pobax.envs.wrappers.gymnax import GymnaxWrapper

def unwrap_env_state(s):
    if hasattr(s, 'env_state'):
        return unwrap_env_state(s.env_state)
    return s


class PixelBraxVecEnvWrapper(GymnaxWrapper):
    def __init__(self, env: VecEnv,
                 size: int = 128,
                 normalize: bool = False):
        super().__init__(env)
        self._env.reset = jax.jit(self._env.reset)
        self._env.step = jax.jit(self._env.step)

        self.renderer = None

        self.normalize = normalize
        self.size = size

    def observation_space(self, params):
        low, high = 0, 255
        if self.normalize:
            high = 1
        return spaces.Box(
            low=low,
            high=high,
            shape=(self.size, self.size, 3),
        )

    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        sys = self._unwrapped._env.sys
        self.renderer = [mujoco.Renderer(sys.mj_model, height=self.size, width=self.size) for _ in range(key.shape[0])]

        _, env_state = self._env.reset(key, params)
        image_obs = self.render(env_state)
        if self.normalize:
            image_obs /= 255.
        return image_obs, env_state

    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            action: Union[int, float],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        _, env_state, reward, done, info = self._env.step(
            key, state, action, params
        )
        image_obs = self.render(env_state)
        if self.normalize:
            image_obs /= 255.
        return image_obs, env_state, reward, done, info

    def render(self, states, mode='rgb_array'):
        states = unwrap_env_state(states)
        # @jax.jit
        # def unpack(s):
        #
        #     n = len(self.renderer)
        #
        #     unpacked = [
        #         jax.tree.map(lambda leaf: leaf[i], s)
        #         for i in range(n)
        #     ]
        #     return unpacked
        #
        # sys = self._unwrapped._env.sys
        # list_states = unpack(states.pipeline_state)

        sys = self._unwrapped._env.sys
        n = len(self.renderer)
        def get_image(state: base.State, i: int):
            d = mujoco.MjData(sys.mj_model)
            d.qpos, d.qvel = state.q[i], state.qd[i]
            mujoco.mj_forward(sys.mj_model, d)
            self.renderer[i].update_scene(d, camera=-1)
            return self.renderer[i].render()

        images = jnp.stack([get_image(states.pipeline_state, i) for i in range(n)])

        return images
