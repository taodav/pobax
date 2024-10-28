from functools import partial
from typing import Optional, Tuple, Union

import chex
import jax
import mujoco
from brax import base
from gymnax.environments import spaces, environment
from jax import numpy as jnp

from pobax.envs import VecEnv
from pobax.envs.jax.tmaze import TMazeState
from pobax.envs.wrappers.gymnax import GymnaxWrapper

def unwrap_env_state(s):
    if hasattr(s, 'env_state'):
        return unwrap_env_state(s.env_state)
    return s

class MjvCamera(mujoco.MjvCamera):  # pylint: disable=missing-docstring

  # Provide this alias for the "type" property for backwards compatibility.
  @property
  def type_(self):
    return self.type

  @type_.setter
  def type_(self, t):
    self.type = t

  @property
  def ptr(self):
    return self


class PixelBraxVecEnvWrapper(GymnaxWrapper):
    def __init__(self, env: VecEnv,
                 size: int = 128,
                 normalize: bool = False,
                 zoom_factor: float = 1.):
        super().__init__(env)
        self._env.reset = jax.jit(self._env.reset)
        self._env.step = jax.jit(self._env.step)

        self.renderer = None

        self.normalize = normalize
        self.size = size
        self.zoom_factor = zoom_factor

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
            camera = MjvCamera()
            camera.fixedcamid = 0
            camera.type = mujoco.mjtCamera.mjCAMERA_FIXED

            self.renderer[i].update_scene(d, camera=camera)
            return self.renderer[i].render()

        images = jnp.stack([get_image(states.pipeline_state, i) for i in range(n)])

        return images




def get_tmaze_image(state: TMazeState, hallway_length: int, size: int):
    in_start = state.grid_idx == 0
    in_junction = state.grid_idx == (hallway_length + 1)
    in_hallway = (1 - in_start) * (1 - in_junction)

    obs = jnp.ones((size, size))
    start_obs_0 = obs * (state.goal_dir * in_start)
    start_obs_1 = obs * ((1 - state.goal_dir) * in_start)
    hallway_obs = obs * in_hallway
    junction_obs = obs * in_junction
    # return jnp.stack((start_obs, hallway_obs, junction_obs), axis=-1)
    return jnp.stack((start_obs_0, start_obs_1, hallway_obs, junction_obs), axis=-1)


class PixelTMazeVecEnvWrapper(PixelBraxVecEnvWrapper):
    def __init__(self, env: VecEnv,
                 size: int = 128,
                 normalize: bool = False):
        super().__init__(env, size=size, normalize=False)
        p_get_tmaze_image = partial(get_tmaze_image, hallway_length=self._env.hallway_length, size=size)
        self.get_tmaze_images = jax.jit(jax.vmap(p_get_tmaze_image))

    def observation_space(self, params):
        low, high = 0, 1
        return spaces.Box(
            low=low,
            high=high,
            shape=(self.size, self.size, 4),
        )

    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        _, env_state = self._env.reset(key, params)
        image_obs = self.render(env_state)
        if self.normalize:
            image_obs /= 255.
        return image_obs, env_state

    def render(self, states, mode='rgb_array'):
        states = states.env_state
        flattened, _ = jax.tree.flatten(states)

        images = self.get_tmaze_images(states)
        # images = jnp.zeros((4, size, size, 3)) + states.grid_idx[0]
        return images


class PixelSimpleChainVecEnvWrapper(PixelBraxVecEnvWrapper):
    def __init__(self, env: VecEnv,
                 size: int = 128,
                 normalize: bool = False):
        super().__init__(env, size=size, normalize=False)

    def observation_space(self, params):
        low, high = 0, 1
        return spaces.Box(
            low=low,
            high=high,
            shape=(self.size, self.size, 2),
        )

    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        _, env_state = self._env.reset(key, params)
        image_obs = self.render(env_state)
        if self.normalize:
            image_obs /= 255.
        return image_obs, env_state

    def render(self, states, mode='rgb_array'):
        return jnp.ones((self.size, self.size, 2))
