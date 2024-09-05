from functools import partial

import chex
import gymnax
from gymnax.environments.environment import Environment, EnvParams
import jax
import jax.numpy as jnp
from jax import random
from typing import Tuple


@chex.dataclass
class TMazeState:
    grid_idx: int
    goal_dir: int


class TMaze(Environment):

    def __init__(self,
                 hallway_length: int = 5,
                 good_reward: float = 4.,
                 bad_reward: float = -0.1,
                 slip_prob: float = 0.0):
        """
        The T-Maze environment, implemented with the Gymnax API.
        You start in grid_idx = 0, which reveals whether the goal is up or down.
        At grid_idx = -2, you can go either up or down. If you go up and goal_up/down and not goal_up,
        then you get the good_reward. Going in the opposite directions at this junction state
        gives you the bad_reward.
        """
        self.hallway_length = hallway_length
        self.good_reward = good_reward
        self.bad_reward = bad_reward
        self.slip_prob = slip_prob

    def observation_space(self, env_params: EnvParams):
        # Good reward, bad reward, hallway, junction and terminal.
        return gymnax.environments.spaces.Box(0, 1, (5, ))

    def action_space(self, env_params: EnvParams):
        """
        Actions are NORTH, SOUTH, EAST, WEST
        """
        return gymnax.environments.spaces.Discrete(4)

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)

    def get_obs(self, state: TMazeState) -> jnp.ndarray:
        obs = jnp.zeros(5)
        start_obs = obs.at[state.goal_dir].set(1)
        hallway_obs = obs.at[2].set(1)
        junction_obs = obs.at[-2].set(1)

        in_start = state.grid_idx == 0
        in_junction = state.grid_idx == (self.hallway_length + 1)
        obs = in_start * start_obs + in_junction * junction_obs + \
              (1 - in_start) * (1 - in_junction) * hallway_obs
        return obs

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: EnvParams) -> Tuple[jnp.ndarray, TMazeState]:
        state = TMazeState(grid_idx=0,
                           goal_dir=random.bernoulli(key).astype(int))
        return self.get_obs(state), state

    def _up_down_transition(self, key: chex.PRNGKey, state: TMazeState, action: int):
        # In here, we assume that action == 0 or 1
        in_junction = state.grid_idx == (self.hallway_length + 1)

        next_idx = state.grid_idx
        if not ((self.slip_prob > 0.) and (random.uniform(key) > self.slip_prob)):
            next_idx = jax.lax.select(in_junction,
                                      state.grid_idx + 1,
                                      state.grid_idx)

        action_up = action == 0
        get_good_reward = action_up * (1 - state.goal_dir) + (1 - action_up) * state.goal_dir
        reward = (get_good_reward * self.good_reward + (1 - get_good_reward) * self.bad_reward) * in_junction
        done = in_junction  # we terminate here b/c we're in the junction and go to terminal state.
        return TMazeState(grid_idx=next_idx, goal_dir=state.goal_dir), reward, done

    def _left_right_transition(self, key: chex.PRNGKey, state: TMazeState, action: int):
        # In here, we assume that action == 2 or 3
        go_right = action == 2
        next_idx = state.grid_idx
        if not ((self.slip_prob > 0.) and (random.uniform(key) > self.slip_prob)):
            next_idx = go_right * jnp.minimum(state.grid_idx + 1, self.hallway_length + 1) + \
                       (1 - go_right) * jnp.maximum(state.grid_idx - 1, 0)
        return TMazeState(grid_idx=next_idx, goal_dir=state.goal_dir), jnp.array(0, dtype=float), False

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self,
                 key: chex.PRNGKey,
                 state: TMazeState,
                 action: int,
                 params: EnvParams):
        next_state, reward, done = jax.lax.cond(action < 2,
                                                self._up_down_transition,
                                                self._left_right_transition,
                                                key, state, action)
        return self.get_obs(next_state), next_state, reward, done, {}

