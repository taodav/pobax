from functools import partial
from typing import Tuple, Optional, Union

import jax
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments import environment, spaces
import navix as nx
from navix.environments import Environment as NavixEnvironment

@struct.dataclass
class NavixState:
    timestep: nx.environments.Timestep


class NavixGymnaxWrapper:
    def __init__(self, env: NavixEnvironment):
        self._env = env

    @property
    def default_params(self):
        return environment.EnvParams(max_steps_in_episode=self._env.max_steps)

    def observation_space(self, params) -> spaces.Box:
        # return spaces.Box(0, 8, (4, 7, 1))
        return self._env.observation_space

    def action_space(self, params) -> spaces.Discrete:
        # Action space here is 3, b/c rest of actions are unused.
        return spaces.Discrete(3)
        # return self._env.action_space

    @partial(jax.jit, static_argnums=(0, -1))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, NavixState]:
        timestep = self._env.reset(key)
        state = NavixState(timestep=timestep)

        return timestep.observation, state

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
            self,
            key: chex.PRNGKey,
            state: NavixState,
            action: int,
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, NavixState, float, bool, dict]:
        timestep = self._env.step(state.timestep, action)
        state = NavixState(timestep=timestep)

        done = jnp.logical_or((timestep.step_type == 1), (timestep.step_type == 2))
        return timestep.observation, state, timestep.reward, done, timestep.info
