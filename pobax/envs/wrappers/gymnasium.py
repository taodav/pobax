from typing import Any, Optional, Tuple, Dict, Union, List

import chex
import jax
import gymnasium as gym
from gymnasium import Wrapper, core
from gymnasium.core import WrapperObsType, WrapperActType, SupportsFloat, Env
from gymnasium.wrappers import AddRenderObservation

from gymnax.environments import environment
from gymnax.environments import spaces


class GymnaxToGymWrapper(gym.Env[core.ObsType, core.ActType]):
    """Wrap Gymnax environment as OOP Gym environment."""

    def __init__(
            self,
            env: environment.Environment,
            params: Optional[environment.EnvParams] = None,
            seed: Optional[int] = None,
            num_envs: Optional[int] = None,
    ):
        """Wrap Gymnax environment as OOP Gym environment.


        Args:
            env: Gymnax Environment instance
            params: If provided, gymnax EnvParams for environment (otherwise uses
              default)
            seed: If provided, seed for JAX PRNG (otherwise picks 0)
        """
        super().__init__()
        self._env = env
        self.env_params = params if params is not None else env.default_params
        self.metadata.update(
            {
                "name": env.name,
                "render_modes": (
                    ["human", "rgb_array"] if hasattr(env, "render") else []
                ),
            }
        )
        self.rng: chex.PRNGKey = jax.random.PRNGKey(0)  # Placeholder
        self._seed(seed)
        self.num_envs = num_envs
        rng = self.rng
        if self.num_envs is not None:
            rng = jax.random.split(self.rng, self.num_envs)
        _, self.env_state = self._env.reset(rng, self.env_params)
        self.max_steps_in_episode = self.env_params.max_steps_in_episode

    @property
    def action_space(self):
        """Dynamically adjust action space depending on params."""
        return spaces.gymnax_space_to_gym_space(self._env.action_space(self.env_params))

    @property
    def observation_space(self):
        """Dynamically adjust state space depending on params."""
        return spaces.gymnax_space_to_gym_space(
            self._env.observation_space(self.env_params)
        )

    def _seed(self, seed: Optional[int] = None):
        """Set RNG seed (or use 0)."""
        self.rng = jax.random.PRNGKey(seed or 0)

    def step(
            self, action: core.ActType
    ) -> Tuple[core.ObsType, float, bool, bool, Dict[Any, Any]]:
        """Step environment, follow new step API."""
        self.rng, step_key = jax.random.split(self.rng)
        step_keys = jax.random.split(step_key, self.num_envs)
        o, self.env_state, r, d, info = self._env.step(
            step_keys, self.env_state, action, self.env_params
        )
        return o, r, d, d, info

    def reset(
            self,
            *,
            seed: Optional[int] = None,
            return_info: bool = False,
            options: Optional[Any] = None,  # dict
    ) -> Tuple[core.ObsType, Any]:  # dict]:
        """Reset environment, update parameters and seed if provided."""
        if seed is not None:
            self._seed(seed)
        if options is not None:
            self.env_params = options.get(
                "env_params", self.env_params
            )  # Allow changing environment parameters on reset
        self.rng, reset_key = jax.random.split(self.rng)
        reset_keys = jax.random.split(reset_key, self.num_envs)
        o, self.env_state = self._env.reset(reset_keys, self.env_params)
        return o, {}

    def render(
            self, mode="human"
    ) -> Optional[Union[core.RenderFrame, List[core.RenderFrame]]]:
        """use underlying environment rendering if it exists, otherwise return None."""
        return getattr(self._env, "render", lambda x, y: None)(
            self.env_state, self.env_params
        )

class PixelOnlyObservationWrapper(AddRenderObservation):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.observation_space['pixels']

    def observation(self, observation):
        dict_observations = super().observation(observation)
        return dict_observations['pixels']


class OnlineReturnsLogWrapper(Wrapper):
    def __init__(self, *args, gamma: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.episode_return = 0
        self.discounted_episode_return = 0
        self.episode_length = 0
        self.returned_episode_return = 0
        self.returned_discounted_episode_return = 0
        self.timestep = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.returned_episode_return = self.episode_return
        self.returned_discounted_episode_return = self.discounted_episode_return
        info = {
            'returned_episode_return': self.returned_episode_return,
            'returned_discounted_episode_return': self.returned_discounted_episode_return,
            'episode_length': self.episode_length,
            **info
        }

        self.episode_return = 0
        self.discounted_episode_return = 0
        self.episode_length = 0
        self.timestep = 0
        return obs, info

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done, truncation, info = self.env.step(action)

        new_episode_return = self.episode_return + reward
        new_discounted_episode_return = self.discounted_episode_return + (self.gamma ** self.episode_length) * reward
        new_episode_length = self.episode_length + 1

        not_done_or_not_trunc = (1 - done) * (1 - truncation)
        self.episode_return = new_episode_return * not_done_or_not_trunc
        self.discounted_episode_return = new_discounted_episode_return * not_done_or_not_trunc
        self.episode_length = new_episode_length * not_done_or_not_trunc

        self.returned_episode_return = self.returned_episode_return * not_done_or_not_trunc \
                                       + new_episode_return * (1 - not_done_or_not_trunc)
        self.returned_discounted_episode_return = self.returned_discounted_episode_return * not_done_or_not_trunc\
                                                  + new_discounted_episode_return * (1 - not_done_or_not_trunc)

        info = {
            'episode_return': self.episode_return,
            'returned_episode_return': self.returned_episode_return,
            'returned_discounted_episode_return': self.returned_discounted_episode_return,
            'episode_length': self.episode_length,
            # **info
        }
        return obs, reward, done, truncation, info

