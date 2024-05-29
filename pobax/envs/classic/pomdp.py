from functools import partial

import chex
from jax import random, jit
import jax.numpy as jnp
from gymnax.environments import environment, spaces


class POMDP(environment.Environment):
    def __init__(self,
                 T: jnp.ndarray,
                 R: jnp.ndarray,
                 p0: jnp.ndarray,
                 gamma: float,
                 phi: jnp.ndarray,
                 fully_observable: bool = False):
        self.gamma = jnp.array(gamma)
        self.T = jnp.array(T)
        self.R = jnp.array(R)
        self.phi = jnp.array(phi)

        self.p0 = jnp.array(p0)
        self.fully_observable = fully_observable

    def observation_space(self, params: environment.EnvParams):
        if self.fully_observable:
            return spaces.Box(0, 1, (self.T.shape[-1],))
        return spaces.Box(0, 1, (self.phi.shape[-1],))

    def action_space(self, params: environment.EnvParams):
        return spaces.Discrete(self.T.shape[0])

    @property
    def default_params(self) -> environment.EnvParams:
        return environment.EnvParams(max_steps_in_episode=1000)

    def get_obs(self, key: chex.PRNGKey, s: jnp.ndarray):
        if self.fully_observable:
            n_states = self.T.shape[-1]
            obs = jnp.zeros(n_states)
            obs = obs.at[s].set(1)
            return obs

        n_obs = self.phi[s].shape[0]

        observed_idx = random.choice(key, n_obs, p=self.phi[s])
        obs = jnp.zeros(n_obs)
        obs = obs.at[observed_idx].set(1)
        return obs

    @partial(jit, static_argnums=(0, ))
    def reset_env(self, key: chex.PRNGKey, params: environment.EnvParams):
        obs_key, init_key = random.split(key)
        state = random.choice(init_key, self.p0.shape[0], p=self.p0)
        return self.get_obs(obs_key, state), state

    @partial(jit, static_argnums=(0, -2))
    def step_env(self,
                 key: chex.PRNGKey,
                 state: jnp.ndarray,
                 action: int,
                 params: environment.EnvParams):
        pr_next_s = self.T[action, state, :]

        next_state_key, obs_key = random.split(key)
        next_state = random.choice(next_state_key, pr_next_s.shape[0], p=pr_next_s)

        reward = self.R[action, state, next_state]

        # Check if next_state is absorbing state
        is_absorbing = (self.T[:, next_state, next_state] == 1)
        terminal = is_absorbing.all() # absorbing for all actions
        observation = self.get_obs(obs_key, next_state)

        return observation, next_state, reward, terminal, {}
