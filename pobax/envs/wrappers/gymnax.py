# taken from https://github.com/luchris429/purejaxrl/blob/main/purejaxrl/wrappers.py
from typing import Optional, Tuple, Union, Callable

from brax import envs
from brax.envs.wrappers.training import EpisodeWrapper, AutoResetWrapper
import chex
from flax import struct
from functools import partial
from gymnax.environments import environment, spaces
import jax
import jax.numpy as jnp
import numpy as np
from sympy.physics.units import action
from brax.base import System
from brax.envs.base import Env
from brax.envs.base import State


class GymnaxWrapper(object):
    """Base class for Gymnax wrappers."""

    def __init__(self, env):
        self._env = env
        if hasattr(env, '_unwrapped'):
            self._unwrapped = env._unwrapped
        else:
            self._unwrapped = env

    # provide proxy access to regular attributes of wrapped object
    def __getattr__(self, name):
        return getattr(self._env, name)


class MaskObservationWrapper(GymnaxWrapper):
    def __init__(self, env: environment.Environment,
                 mask_dims: list,
                 **kwargs):
        super().__init__(env)
        self.mask_dims = jnp.array(mask_dims, dtype=int)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        low = self._env.observation_space(params).low
        if isinstance(low, jnp.ndarray):
            low = low[self.mask_dims]

        high = self._env.observation_space(params).high
        if isinstance(high, jnp.ndarray):
            high = high[self.mask_dims]

        return spaces.Box(
            low=low,
            high=high,
            shape=(self.mask_dims.shape[0],),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,-1))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = obs[self.mask_dims]
        return obs, state

    @partial(jax.jit, static_argnums=(0,-1))
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            action: Union[int, float],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = obs[self.mask_dims]
        return obs, state, reward, done, info


class FlattenObservationWrapper(GymnaxWrapper):
    """Flatten the observations of the environment."""

    def __init__(self, env: environment.Environment):
        super().__init__(env)

    def observation_space(self, params) -> spaces.Box:
        assert isinstance(
            self._env.observation_space(params), spaces.Box
        ), "Only Box spaces are supported for now."
        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=(np.prod(self._env.observation_space(params).shape),),
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,-1))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state

    @partial(jax.jit, static_argnums=(0,-1))
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            action: Union[int, float],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        obs = jnp.reshape(obs, (-1,))
        return obs, state, reward, done, info


@struct.dataclass
class LogEnvState:
    env_state: environment.EnvState
    episode_returns: float
    discounted_episode_returns: float
    episode_lengths: int
    returned_episode_returns: float
    returned_discounted_episode_returns: float
    returned_episode_lengths: int
    timestep: int


class LogWrapper(GymnaxWrapper):
    """Log the episode returns and lengths."""

    def __init__(self, env: environment.Environment, gamma: float = 0.99):
        super().__init__(env)
        self.gamma = gamma

    @partial(jax.jit, static_argnums=(0, -1))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        state = LogEnvState(env_state, 0.0, 0.0, 0, 0.0, 0.0, 0, 0)
        return obs, state

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            action: Union[int, float],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        new_episode_return = state.episode_returns + reward
        new_discounted_episode_return = state.discounted_episode_returns + (self.gamma ** state.episode_lengths) * reward
        new_episode_length = state.episode_lengths + 1

        state = LogEnvState(
            env_state=env_state,
            episode_returns=new_episode_return * (1 - done),
            discounted_episode_returns=new_discounted_episode_return * (1 - done),
            episode_lengths=new_episode_length * (1 - done),
            returned_episode_returns=state.returned_episode_returns * (1 - done)
                                     + new_episode_return * done,
            returned_discounted_episode_returns=state.returned_discounted_episode_returns * (1 - done)
                                                + new_discounted_episode_return * done,
            returned_episode_lengths=state.returned_episode_lengths * (1 - done)
                                     + new_episode_length * done,
            timestep=state.timestep + 1,
        )
        info["returned_episode_returns"] = state.returned_episode_returns
        info["returned_discounted_episode_returns"] = state.returned_discounted_episode_returns
        info["returned_episode_lengths"] = state.returned_episode_lengths
        info["timestep"] = state.timestep
        info["returned_episode"] = done
        info["reward"] = reward
        return obs, state, reward, done, info


class BraxGymnaxWrapper:
    def __init__(self, env_name, backend="positional"):
        env = envs.get_environment(env_name=env_name, backend=backend)
        self.max_steps_in_episode = 1000
        env = EpisodeWrapper(env, episode_length=self.max_steps_in_episode, action_repeat=1)
        env = AutoResetWrapper(env)
        self.name = env_name
        self._env = env
        self.action_size = env.action_size
        self.observation_size = (env.observation_size,)

    def reset(self, key, params=None):
        state = self._env.reset(key)
        return state.obs, state

    def step(self, key, state, action, params=None):
        next_state = self._env.step(state, action)
        return next_state.obs, next_state, next_state.reward, next_state.done > 0.5, {}

    def observation_space(self, params):
        return spaces.Box(
            low=-jnp.inf,
            high=jnp.inf,
            shape=(self._env.observation_size,),
        )

    def action_space(self, params):
        return spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self._env.action_size,),
        )


@struct.dataclass
class CraftEnvParams:
    max_steps_in_episode: int = 1
    craft_env_params: environment.EnvParams = None


class CraftaxGymnaxWrapper:
    def __init__(self, env_name):
        from craftax.craftax_env import make_craftax_env_from_name
        env = make_craftax_env_from_name(env_name, auto_reset=True)
        self.max_steps_in_episode = 100000
        self.name = env_name
        self._env = env
        self.env_params = CraftEnvParams(max_steps_in_episode=self.max_steps_in_episode, craft_env_params=env.default_params)

    @partial(jax.jit, static_argnums=(0,-1))
    def reset(self, key, params=None):
        obs, state = self._env.reset_env(key, params.craft_env_params)
        return obs, state

    @partial(jax.jit, static_argnums=(0,-1))
    def step(self, key, state, action, params=None):
        # Pixel value is already normalized
        next_obs, next_state, reward, done, info = self._env.step_env(key, state, action, params.craft_env_params)
        return next_obs, next_state, reward, done, {}
        

    def observation_space(self, params):
        return self._env.observation_space(params.craft_env_params)

    def action_space(self, params):
        return self._env.action_space(params.craft_env_params)


class ClipAction(GymnaxWrapper):
    def __init__(self, env, low=-1.0, high=1.0):
        super().__init__(env)
        self.low = low
        self.high = high

    def step(self, key, state, action, params=None):
        """TODO: In theory the below line should be the way to do this."""
        # action = jnp.clip(action, self.env.action_space.low, self.env.action_space.high)
        action = jnp.clip(action, self.low, self.high)
        return self._env.step(key, state, action, params)

class AutoResetEnvWrapper(GymnaxWrapper):
    """Provides standard auto-reset functionality, providing the same behaviour as Gymnax-default."""

    def __init__(self, env):
        super().__init__(env)

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, key, params=None):
        return self._env.reset(key, params)

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        obs_st, state_st, reward, done, info = self._env.step(
            _rng, state, action, params
        )

        rng, _rng = jax.random.split(rng)
        obs_re, state_re = self._env.reset(_rng, params)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return obs, state

        obs, state = auto_reset(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info

class TransformObservation(GymnaxWrapper):
    def __init__(self, env, transform_obs):
        super().__init__(env)
        self.transform_obs = transform_obs

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        return self.transform_obs(obs), state

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.transform_obs(obs), state, reward, done, info


class TransformReward(GymnaxWrapper):
    def __init__(self, env, transform_reward):
        super().__init__(env)
        self.transform_reward = transform_reward

    def step(self, key, state, action, params=None):
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return obs, state, self.transform_reward(reward), done, info


class VecEnv(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)
        self.reset = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))


@struct.dataclass
class NormalizeVecObsEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    env_state: environment.EnvState


class NormalizeVecObservation(GymnaxWrapper):
    def __init__(self, env):
        super().__init__(env)

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        state = NormalizeVecObsEnvState(
            mean=jnp.zeros_like(obs),
            var=jnp.ones_like(obs),
            count=1e-4,
            env_state=state,
        )
        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=state.env_state,
        )

        return (obs - state.mean) / jnp.sqrt(state.var + 1e-8), state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )

        batch_mean = jnp.mean(obs, axis=0)
        batch_var = jnp.var(obs, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecObsEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            env_state=env_state,
        )
        return (
            (obs - state.mean) / jnp.sqrt(state.var + 1e-8),
            state,
            reward,
            done,
            info,
        )


@struct.dataclass
class NormalizeVecRewEnvState:
    mean: jnp.ndarray
    var: jnp.ndarray
    count: float
    return_val: float
    env_state: environment.EnvState


class NormalizeVecReward(GymnaxWrapper):
    def __init__(self, env, gamma):
        super().__init__(env)
        self.gamma = gamma

    def reset(self, key, params=None):
        obs, state = self._env.reset(key, params)
        batch_count = obs.shape[0]
        state = NormalizeVecRewEnvState(
            mean=0.0,
            var=1.0,
            count=1e-4,
            return_val=jnp.zeros((batch_count,)),
            env_state=state,
        )
        return obs, state

    def step(self, key, state, action, params=None):
        obs, env_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        return_val = state.return_val * self.gamma * (1 - done) + reward

        batch_mean = jnp.mean(return_val, axis=0)
        batch_var = jnp.var(return_val, axis=0)
        batch_count = obs.shape[0]

        delta = batch_mean - state.mean
        tot_count = state.count + batch_count

        new_mean = state.mean + delta * batch_count / tot_count
        m_a = state.var * state.count
        m_b = batch_var * batch_count
        M2 = m_a + m_b + jnp.square(delta) * state.count * batch_count / tot_count
        new_var = M2 / tot_count
        new_count = tot_count

        state = NormalizeVecRewEnvState(
            mean=new_mean,
            var=new_var,
            count=new_count,
            return_val=return_val,
            env_state=env_state,
        )
        return obs, state, reward / jnp.sqrt(state.var + 1e-8), done, info


class ActionConcatWrapper(GymnaxWrapper):
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

    def observation_space(self, params) -> spaces.Box:
        og_obs_space_shape = self._env.observation_space(params).shape

        if len(og_obs_space_shape) == 1:
            shape = (og_obs_space_shape[0] + self.action_size(params),)
        elif len(og_obs_space_shape) == 3:
            # images
            shape = (og_obs_space_shape[0], og_obs_space_shape[1], og_obs_space_shape[2] + self.action_size(params))
        else:
            raise NotImplementedError

        return spaces.Box(
            low=self._env.observation_space(params).low,
            high=self._env.observation_space(params).high,
            shape=shape,
            dtype=self._env.observation_space(params).dtype,
        )

    @partial(jax.jit, static_argnums=(0,-1))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        action_vec = jnp.zeros(self.action_size(params))
        obs, state = self._env.reset(key, params)
        obs_shape = self.observation_space(params).shape
        if len(obs_shape) == 1:
            obs = jnp.concatenate([obs, action_vec])

        elif len(obs_shape) == 3:
            action_vec = action_vec[None, None, ...]
            h, w, c = obs_shape
            action_img = action_vec.repeat(h, axis=0).repeat(w, axis=1)
            obs = jnp.concatenate([obs, action_img], axis=-1)
        else:
            raise NotImplementedError

        return obs, state

    @partial(jax.jit, static_argnums=(0,-1))
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            action: Union[int, float, jnp.ndarray],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        action_vec = action
        action_space = self.action_space(params)
        if isinstance(action_space, spaces.Discrete):
            action_vec = jnp.eye(action_space.n)[action]

        obs_shape = self.observation_space(params).shape
        if len(obs_shape) == 1:
            obs = jnp.concatenate([obs, action_vec])

        elif len(obs_shape) == 3:
            action_vec = action_vec[None, None, ...]
            h, w, c = obs_shape
            action_img = action_vec.repeat(h, axis=0).repeat(w, axis=1)
            
            obs = jnp.concatenate([obs, action_img], axis=-1)
        else:
            raise NotImplementedError
        return obs, state, reward, done, info


class OptimisticResetVecEnvWrapper(GymnaxWrapper):
    """
    Provides efficient 'optimistic' resets.
    The wrapper also necessarily handles the batching of environment steps and resetting.
    reset_ratio: the number of environment workers per environment reset.  Higher means more efficient but a higher
    chance of duplicate resets.
    """

    def __init__(self, env, num_envs: int, reset_ratio: int):
        super().__init__(env)

        self.num_envs = num_envs
        self.reset_ratio = reset_ratio
        assert (
            num_envs % reset_ratio == 0
        ), "Reset ratio must perfectly divide num envs."
        self.num_resets = self.num_envs // reset_ratio

        self.reset_fn = jax.vmap(self._env.reset, in_axes=(0, None))
        self.step_fn = jax.vmap(self._env.step, in_axes=(0, 0, 0, None))

    @partial(jax.jit, static_argnums=(0, 2))
    def reset(self, rng, params=None):
        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs, env_state = self.reset_fn(rngs, params)
        return obs, env_state

    @partial(jax.jit, static_argnums=(0, 4))
    def step(self, rng, state, action, params=None):

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_envs)
        obs_st, state_st, reward, done, info = self.step_fn(rngs, state, action, params)

        rng, _rng = jax.random.split(rng)
        rngs = jax.random.split(_rng, self.num_resets)
        obs_re, state_re = self.reset_fn(rngs, params)

        rng, _rng = jax.random.split(rng)
        reset_indexes = jnp.arange(self.num_resets).repeat(self.reset_ratio)

        being_reset = jax.random.choice(
            _rng,
            jnp.arange(self.num_envs),
            shape=(self.num_resets,),
            p=done,
            replace=False,
        )
        reset_indexes = reset_indexes.at[being_reset].set(jnp.arange(self.num_resets))

        obs_re = obs_re[reset_indexes]
        state_re = jax.tree_map(lambda x: x[reset_indexes], state_re)

        # Auto-reset environment based on termination
        def auto_reset(done, state_re, state_st, obs_re, obs_st):
            state = jax.tree_map(
                lambda x, y: jax.lax.select(done, x, y), state_re, state_st
            )
            obs = jax.lax.select(done, obs_re, obs_st)

            return state, obs

        state, obs = jax.vmap(auto_reset)(done, state_re, state_st, obs_re, obs_st)

        return obs, state, reward, done, info

def _identity_randomization_fn(
    sys: System, num_worlds: int
) -> Tuple[System, System]:
  """Tile the necessary axes for the Madrona BatchRenderer."""
  in_axes = jax.tree_util.tree_map(lambda x: 0, sys)
  sys = jax.tree_util.tree_map(lambda x: jnp.repeat(jnp.expand_dims(x, 0), num_worlds, axis=0), sys)
  return sys, in_axes

# Inspired from https://github.com/shacklettbp/madrona_mjx/blob/main/src/madrona_mjx/wrapper.py
class MadronaWrapper(GymnaxWrapper):
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
    self.sys = self._unwrapped._env.sys
    self._sys_v, self._in_axes = randomization_fn(self.sys)

  def _env_fn(self, sys: System) -> Env:
    env = self._env
    env._unwrapped._env.sys = sys
    return env

  def reset(self, rng: jax.Array, params: Optional[environment.EnvParams]=None) -> State:
    def reset(sys, rng, params):
      env = self._env_fn(sys=sys)
      return env.reset(rng, params)

    obs, state = jax.vmap(reset, in_axes=[self._in_axes, 0, None])(self._sys_v, rng, params)
    return obs, state

  def step(self, key, state: State, action: jax.Array, params: Optional[environment.EnvParams]=None) -> State:
    def step(sys, key, s, a, params):
      env = self._env_fn(sys=sys)
      return env.step(key, s, a, params)

    res = jax.vmap(step, in_axes=[self._in_axes, 0, 0, 0, None])(
        self._sys_v, key, state, action, params
    )
    return res