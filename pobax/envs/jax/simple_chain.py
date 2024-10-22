from functools import partial
from typing import Any

import chex
import jax
import jax.numpy as jnp
from gymnax.environments.environment import Environment, EnvParams
from gymnax.environments import environment, spaces
import numpy as np

from pobax.envs.wrappers.gymnax import GymnaxWrapper

@chex.dataclass
class SimpleChainState:
    pos_idx: chex.Array


class SimpleChain(Environment):
    def __init__(self,
                 n: int = 10,
                 reward_in_obs: bool = False
                 ):
        """
        Simple func. approx single chain. Always returns an observation of 1.
        :param n: length of chain
        """
        self.n = n
        self.reward_in_obs = reward_in_obs

    def observation_space(self, params: EnvParams):
        n_obs = 1
        if self.reward_in_obs:
            n_obs += 1

        return spaces.Box(0, 1, (n_obs, ))

    def action_space(self, params: EnvParams):
        return spaces.Discrete(1)

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(
            self, key: chex.PRNGKey, params: EnvParams
    ) -> tuple[chex.Array, SimpleChainState]:
        state = SimpleChainState(pos_idx=jnp.array(0, dtype=int))
        return self.get_obs(state), state

    @partial(jax.jit, static_argnums=(0,))
    def get_reward(self, state: SimpleChainState) -> jnp.ndarray:
        return (state.pos_idx == self.n).astype(float)

    @partial(jax.jit, static_argnums=(0,))
    def get_obs(self, state: SimpleChainState) -> jnp.ndarray:
        if self.reward_in_obs:
            return jnp.array([1, self.get_reward(state)])
        return jnp.array([1])

    @partial(jax.jit, static_argnums=(0,))
    def get_terminal(self, state: SimpleChainState) -> bool:
        return state.pos_idx == self.n

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: SimpleChainState,
        action: int,
        params: EnvParams,
    ) -> tuple[chex.Array, SimpleChainState, jnp.ndarray, jnp.ndarray, dict]:
        new_state = SimpleChainState(pos_idx=jnp.minimum(state.pos_idx + 1, jnp.array(self.n)))
        return self.get_obs(new_state), new_state, self.get_reward(new_state), self.get_terminal(new_state), {}


class FullyObservableSimpleChain(SimpleChain):
    def __init__(self, n: int = 10, reward_in_obs: bool = False):
        super().__init__(n=n, reward_in_obs=reward_in_obs)

    def observation_space(self, params: EnvParams):
        n_obs = self.n
        if self.reward_in_obs:
            n_obs += 1

        return spaces.Box(0, 1, (n_obs, ))

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        return self.state.copy()
