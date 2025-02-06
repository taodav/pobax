import functools
from functools import partial
from typing import Optional, Tuple, Union, Callable

import chex
import jax
import mujoco
from brax import base
from gymnax.environments import spaces, environment
from jax import numpy as jnp

from pobax.envs import VecEnv
from pobax.envs.jax.tmaze import TMazeState
from pobax.envs.wrappers.gymnax import GymnaxWrapper

from brax.base import System
from brax.envs.base import Env
from brax.envs.base import State
from brax.envs.base import Wrapper
from etils import epath

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
        sys = self._unwrapped._env.sys
        print(type(sys.mj_model))
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

class PixelCraftaxVecEnvWrapper(GymnaxWrapper):
    def __init__(self, env: VecEnv,
                 normalize: bool = False):
        super().__init__(env)
        self._env.reset = jax.jit(self._env.reset)
        self._env.step = jax.jit(self._env.step)

        self.renderer = None

        self.normalize = normalize
        self.size = 110

    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        image_obs, env_state = self._env.reset(key, params)
        # Craftax already returned normalized visual input
        image_obs = self.get_obs(image_obs, self.normalize)
        return image_obs, env_state
    
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            action: Union[int, float],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        image_obs, env_state, reward, done, info = self._env.step(
            key, state, action, params
        )
        image_obs = self.get_obs(image_obs, self.normalize)
        return image_obs, env_state, reward, done, info

    def get_obs(self, obs, normalize):
        if not normalize:
            obs *= 255
        assert len(obs.shape) == 4
        assert obs.shape[1] == 130
        obs = obs[:,:90, :, :]
        return obs
    
    def observation_space(self, params):
        low, high = 0, 255
        if self.normalize:
            high = 1
        return spaces.Box(
            low=low,
            high=high,
            shape=(110, 90, 3),
        )


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
    

"""Custom wrappers that extend Brax wrappers"""

def load_model(path: str):
  path = epath.Path(path)
  xml = path.read_text()
  assets = {}
  for f in path.parent.glob('*.xml'):
    assets[f.name] = f.read_bytes()
    for f in (path.parent / 'assets').glob('*'):
      assets[f.name] = f.read_bytes()
  model = mujoco.MjModel.from_xml_string(xml, assets)
  return model


def _identity_randomization_fn(
    sys: System, num_worlds: int
) -> Tuple[System, System]:
  """Tile the necessary axes for the Madrona BatchRenderer."""
  in_axes = jax.tree_util.tree_map(lambda x: None, sys)
  in_axes = in_axes.tree_replace({
      'geom_rgba': 0,
      'geom_matid': 0,
      'geom_size': 0,
      'light_pos': 0,
      'light_dir': 0,
      'light_directional': 0,
      'light_castshadow': 0,
      'light_cutoff': 0,
  })

  sys = sys.tree_replace({
      'geom_rgba': jnp.repeat(
          jnp.expand_dims(sys.geom_rgba, 0), num_worlds, axis=0
      ),
      'geom_matid': jnp.repeat(
          jnp.expand_dims(sys.geom_matid, 0), num_worlds, axis=0
      ),
      'geom_size': jnp.repeat(
          jnp.expand_dims(sys.geom_size, 0), num_worlds, axis=0
      ),
      'light_pos': jnp.repeat(
          jnp.expand_dims(sys.light_pos, 0), num_worlds, axis=0
      ),
      'light_dir': jnp.repeat(
          jnp.expand_dims(sys.light_dir, 0), num_worlds, axis=0
      ),
      'light_directional': jnp.repeat(
          jnp.expand_dims(sys.light_directional, 0), num_worlds, axis=0
      ),
      'light_castshadow': jnp.repeat(
          jnp.expand_dims(sys.light_castshadow, 0), num_worlds, axis=0
      ),
      'light_cutoff': jnp.repeat(
          jnp.expand_dims(sys.light_cutoff, 0), num_worlds, axis=0
      ),
  })

  return sys, in_axes


class MadronaWrapper(Wrapper):
  """Wrapper to Vmap an environment that uses the Madrona BatchRenderer.

  Madrona expects certain MjModel axes to be batched so that the buffers can
  be copied to the GPU. Therefore we need to dummy batch the model to create
  the correct sized buffers for those not using randomization functions,
  and for those using randomization we ensure the correct axes are batched.

  Use this instead of the Brax VmapWrapper and DomainRandimzationWrapper."""

  def __init__(
      self,
      env: Env,
      num_worlds,
      randomization_fn: Optional[
          Callable[[System], Tuple[System, System]]
      ] = None,
  ):
    super().__init__(env)
    self.num_worlds = num_worlds
    if not randomization_fn:
      randomization_fn = functools.partial(
          _identity_randomization_fn, num_worlds=num_worlds
      )

    self._sys_v, self._in_axes = randomization_fn(self.sys)
    # For user-made DR functions, ensure that the output model includes the
    # needed in_axes and has the correct shape for madrona initialization.
    required_fields = [
        'geom_rgba',
        'geom_matid',
        'geom_size',
        'light_pos',
        'light_dir',
        'light_directional',
        'light_castshadow',
        'light_cutoff',
    ]
    for field in required_fields:
      assert hasattr(self._env._in_axes, field), f'{field} not in in_axes'
      assert (
          getattr(self._env._mjx_model_v, field).shape[0] == num_worlds
      ), f'{field} shape does not match num_worlds'

  def _env_fn(self, sys: System) -> Env:
    env = self.env
    env.unwrapped.sys = sys
    return env

  def reset(self, rng: jax.Array) -> State:
    def reset(sys, rng):
      env = self._env_fn(sys=sys)
      return env.reset(rng)

    state = jax.vmap(reset, in_axes=[self._in_axes, 0])(self._sys_v, rng)
    return state

  def step(self, state: State, action: jax.Array) -> State:
    def step(sys, s, a):
      env = self._env_fn(sys=sys)
      return env.step(s, a)

    res = jax.vmap(step, in_axes=[self._in_axes, 0, 0])(
        self._sys_v, state, action
    )
    return res
