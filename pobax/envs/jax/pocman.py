from functools import partial
from typing import Tuple, Union, Optional

import chex
import gymnax
from chex import PRNGKey
from gymnax.environments.environment import EnvParams, Environment
from gymnax.environments import environment
import jax
import jax.numpy as jnp
from jumanji.environments.routing.pac_man import PacMan, State
from jumanji.environments.routing.pac_man.generator import AsciiGenerator
import numpy as np

from pobax.utils.grid import agent_centric_map
from pobax.envs.wrappers.gymnax import GymnaxWrapper

# It's important for [0, 0] to be an unreachable spot,
# for pellet tracking.
SMALLER_GAME_MAP = [
    'XXXXXXXXXXXXXXXXXXX',
    'X  S           S  X',
    'X XX XXX X XXX XX X',
    'XO               OX',
    'X XX X XXXXX X XX X',
    'X    X  TXT  X    X',
    'XXXX XXX X XXX XXXX',
    'XXXX XT G G TX XXXX',
    'XXXX X X   X X XXXX',
    '     X XG GX X     ',
    'XXXX X XXXXX X XXXX',
    'XXXX X       X XXXX',
    'XXXX X XXXXX X XXXX',
    'X                 X',
    'X XX XXX X XXX XX X',
    'XO X S   P   S X OX',
    'XX X X XXXXX X X XX',
    'X    X   X   X    X',
    'X XXXXXX X XXXXXX X',
    'X                 X',
    'XXXXXXXXXXXXXXXXXXX'
]


def generate_los_map(generator: AsciiGenerator):
    """
    Generate a line-of-sight map for each reachable space in the environment.
    """
    space = np.array(generator.numpy_maze)
    los_map = np.zeros((space.shape[0], space.shape[1], 4, space.shape[0], space.shape[1]))
    for coords in generator.reachable_spaces:
        # x is row, y is column
        y, x = coords
        mask = np.zeros_like(generator.numpy_maze)
        mask[x, y] = 1.

        # North
        north_mask = mask.copy()
        all_north_space = space[:x, y].copy()
        if all_north_space.shape[0] > 0:
            north_mask_begin_idx = np.argmin(np.flip(all_north_space))
            if north_mask_begin_idx > 0:
                all_north_space[:-north_mask_begin_idx] = 0
                north_mask[:x, y] = all_north_space

        south_mask = mask.copy()
        all_south_space = space[x + 1:, y].copy()
        if all_south_space.shape[0] > 0:
            south_mask_begin_idx = np.argmin(all_south_space)
            all_south_space[south_mask_begin_idx:] = 0
            south_mask[x + 1:, y] = all_south_space

        east_mask = mask.copy()
        all_east_space = space[x, y + 1:].copy()
        if all_east_space.shape[0] > 0:
            east_mask_begin_idx = np.argmin(all_east_space)
            all_east_space[east_mask_begin_idx:] = 0
            east_mask[x, y + 1:] = all_east_space

        west_mask = mask.copy()
        all_west_space = space[x, :y].copy()
        if all_west_space.shape[0] > 0:
            west_mask_begin_idx = np.argmin(np.flip(all_west_space))
            if west_mask_begin_idx > 0:
                all_west_space[:-west_mask_begin_idx] = 0
                west_mask[x, :y] = all_west_space

        los_map[x, y] = np.stack([north_mask, west_mask, south_mask, east_mask])

    return jnp.array(los_map)


@chex.dataclass
class PerfectMemoryPocmanState:
    env_state: State
    wall_map: jnp.ndarray
    ghost_map: jnp.ndarray
    pellet_map: jnp.ndarray


class PerfectMemoryWrapper(GymnaxWrapper):
    def __init__(self, env, ghost_obs_decay_rate: float = 0.9):
        super().__init__(env)
        self.ghost_obs_decay_rate = ghost_obs_decay_rate

    def observation_space(self, params: EnvParams):
        return gymnax.environments.spaces.Box(0, 1, (self._env.x_size, self._env.y_size, 4))

    @staticmethod
    def _update_maps(prev_wall_map: jnp.ndarray,
                     prev_pellet_map: jnp.ndarray,
                     prev_ghost_map: jnp.ndarray,
                     obs: jnp.ndarray):
        pos_row, pos_col = prev_wall_map.shape[0] // 2, prev_wall_map.shape[1] // 2

        # Walls and pellets
        positions = jnp.array([[pos_row - 1, pos_col],
                               [pos_row, pos_col + 1],
                               [pos_row + 1, pos_col],
                               [pos_row, pos_col - 1],
                               [pos_row, pos_col]])
        wall_map = prev_wall_map.at[positions[:, 0], positions[:, 1]].set(jnp.concatenate([obs[:4], jnp.array([1])]))
        pellet_map = prev_pellet_map.at[positions[:, 0], positions[:, 1]].set(obs[4])

        # Ghosts
        x, y = jnp.ogrid[:prev_ghost_map.shape[0], :prev_ghost_map.shape[1]]
        distance = jnp.abs(x - pos_row) + jnp.abs(y - pos_col)
        ghost_map = prev_ghost_map.at[distance <= 2].set(obs[5])
        ghost_map = ghost_map.at[pos_row, :pos_col].set(obs[6])
        ghost_map = ghost_map.at[:pos_row, pos_col].set(obs[7])
        ghost_map = ghost_map.at[pos_row, pos_col + 1:].set(obs[8])
        ghost_map = ghost_map.at[pos_row + 1:, pos_col].set(obs[9])

        powerpill = jnp.zeros_like(prev_wall_map) + obs[10]

        return wall_map, ghost_map, pellet_map, powerpill

    @partial(jax.jit, static_argnums=(0,))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, env_state = self._env.reset(key, params)
        rows, cols = env_state.grid.shape
        expanded_map = jnp.zeros((2 * rows - 1, 2 * cols - 1)) - 1

        wall_map, ghost_map, pellet_map, powerpill = \
            self._update_maps(expanded_map, expanded_map, expanded_map, obs)

        obs = jnp.stack([wall_map, ghost_map, pellet_map, powerpill], axis=-1)

        return (obs,
                PerfectMemoryPocmanState(
                    env_state=env_state,
                    wall_map=wall_map,
                    ghost_map=ghost_map,
                    pellet_map=pellet_map
                ))

    @partial(jax.jit, static_argnums=(0,))
    def step(
            self,
            key: chex.PRNGKey,
            state: PerfectMemoryPocmanState,
            action: Union[int, float, jnp.ndarray],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, PerfectMemoryPocmanState, float, bool, dict]:
        obs, next_rs_state, reward, done, info = self._env.step(
            key, state.env_state, action, params
        )
        rows, cols = next_rs_state.grid.shape
        expanded_map = jnp.zeros((2 * rows - 1, 2 * cols - 1)) - 1

        # first we shift our map depending on movement
        def shift_map(arr: jnp.ndarray):
            if action == 0:  # UP
                shifted = expanded_map.at[1:].set(arr[:-1])
            elif action == 1:  # LEFT
                shifted = expanded_map.at[:, 1:].set(arr[:, :-1])
            elif action == 2:  # DOWN
                shifted = expanded_map.at[:-1].set(arr[1:])
            elif action == 3:  # RIGHT
                shifted = expanded_map.at[:, :-1].set(arr[:, 1:])
            return shifted

        shifted_wall_map, shifted_ghost_map, shifted_pellet_map = \
            jax.vmap(shift_map)(jnp.stack([state.wall_map, state.ghost_map, state.pellet_map]))

        # now we need to decay our ghost map, seeing as they move
        shifted_ghost_map *= self.ghost_obs_decay_rate

        # TODO: now we incorporate our current obs
        wall_map, ghost_map, pellet_map, powerpill = \
            self._update_maps(shifted_wall_map, shifted_pellet_map, shifted_ghost_map, obs)

        obs = jnp.stack([wall_map, ghost_map, pellet_map, powerpill], axis=-1)

        next_state = PerfectMemoryPocmanState(
            env_state=next_rs_state,
            wall_map=wall_map,
            ghost_map=ghost_map,
            pellet_map=pellet_map
        )
        return obs, next_state, reward, done, info


class PocManStateWrapper(GymnaxWrapper):

    def observation_space(self, params: EnvParams):
        # one_side = 2 * max(self._env.x_size, self._env.y_size) - 1
        # return gymnax.environments.spaces.Box(0, 1, (one_side, one_side, 2))
        return gymnax.environments.spaces.Box(0, 1, (self._env.x_size, self._env.y_size, 6))

    def get_obs(self, state: State) -> jnp.ndarray:
        """
        For get_obs, we need a few channels:
        player positions
        ghost positions
        pellet occupancy
        """
        pellet_locs, power_up_locs = state.pellet_locations, state.power_up_locations
        ghost_locs, old_ghost_locs = state.ghost_locations, state.old_ghost_locations

        # grid with -1 for walls
        grid = state.grid * (-1)

        # +1 for pellets
        grid_with_pellets = grid.at[pellet_locs[:, 0], pellet_locs[:, 1]].set(1)

        # +1 for power up available
        grid_with_power_up = grid.at[power_up_locs[:, 0], power_up_locs[:, 1]].set(1)

        # position
        position_grid = jnp.zeros_like(grid).at[state.player_locations[0], state.player_locations[1]].set(1)

        # separate channel for ghost locs
        ghost_position_grid = jnp.zeros_like(grid).at[ghost_locs[:, 0], ghost_locs[:, 1]].set(1)
        old_ghost_position_grid = jnp.zeros_like(grid).at[old_ghost_locs[:, 0], old_ghost_locs[:, 1]].set(1)

        ghost_grid = ((state.frightened_state_time == 0) * (-1) * ghost_position_grid + \
                      (state.frightened_state_time != 0) * state.frightened_state_time * ghost_position_grid)

        last_direction = jnp.ones_like(grid) * state.last_direction

        # stack em
        combined_grids = jnp.stack((grid_with_pellets, grid_with_power_up, ghost_grid, old_ghost_position_grid, position_grid, last_direction), axis=-1)

        # always have player at center
        # position_encoded_grids = agent_centric_map(combined_grids, state.player_locations, jnp.array(3))
        # return position_encoded_grids
        return combined_grids

    @partial(jax.jit, static_argnums=(0,-1))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        _, state = self._env.reset(key, params)
        obs = self.get_obs(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            action: Union[int, float, jnp.ndarray],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        _, state, reward, done, info = self._env.step(
            key, state, action, params
        )
        obs = self.get_obs(state)

        return obs, state, reward, done, info


class PocMan(PacMan, Environment):
    """
    Partially observable PacMan.
    NOTE:
        state.player_locations.x == row player is in,
        state.player_locations.y == col player is in.
        EVERYTHING ELSE IN STATE (and self.generator) is in (y, x),
        so the normal cartesian coordinates ordering
        (column, row).
    """
    def __init__(self):
        generator = AsciiGenerator(SMALLER_GAME_MAP)
        super().__init__(generator=generator)

        # generate line-of-sight masks for each possible position.
        self.line_sight_map = generate_los_map(generator)
        self.gamma = 0.95

    @property
    def default_params(self):
        return EnvParams(max_steps_in_episode=self.time_limit)

    def observation_space(self, params: EnvParams):
        return gymnax.environments.spaces.Box(0, 1, (11,))

    def action_space(self, params: EnvParams):
        return gymnax.environments.spaces.Discrete(4)

    def get_obs(self, state: State) -> jnp.ndarray:
        # loc is used for comparisons in everything else in the
        # state dict
        loc = jnp.array([state.player_locations.y, state.player_locations.x])
        obs = jnp.zeros(11, dtype=int)

        # First calculate walls around the agent obs[:4]
        obs = obs.at[0].set(state.grid[jnp.maximum(state.player_locations.x - 1, 0), state.player_locations.y])  # up
        obs = obs.at[1].set(state.grid[state.player_locations.x, jnp.minimum(state.player_locations.y + 1, state.grid.shape[1] - 1)])  # right
        obs = obs.at[2].set(state.grid[jnp.minimum(state.player_locations.x + 1, state.grid.shape[0] - 1), state.player_locations.x])  # down
        obs = obs.at[3].set(state.grid[state.player_locations.x, jnp.maximum(state.player_locations.y - 1, 0)])  # left

        # Now calculate if you smell food,
        # manhattan dist <= 1 means obs[4] = 1.
        pellet_dists = jnp.abs(loc[None, ...] - state.pellet_locations).sum(axis=-1)
        obs = obs.at[4].set(jnp.any(pellet_dists <= 1).astype(obs.dtype))

        # Now calculate if you can hear a ghost
        # manhattan dist <= 2 means obs[5]
        ghost_dists = jnp.abs(loc[None, ...] - state.ghost_locations).sum(axis=-1)
        obs = obs.at[5].set(jnp.any(ghost_dists <= 2).astype(obs.dtype))

        # Now LOS calculations
        # if we can see a ghost in each direction, obs[6:10]
        los_grid = self.line_sight_map[state.player_locations.x, state.player_locations.y]
        ghosts_grid = jnp.zeros_like(state.grid).at[state.ghost_locations[:, 1], state.ghost_locations[:, 0]].set(1)
        any_in_dirs = (los_grid * ghosts_grid[None, ...]).sum(axis=-1).sum(axis=-1).astype(obs.dtype)
        obs = obs.at[6:10].set(any_in_dirs)

        # If we have a powerpill, obs[10]
        obs = obs.at[10].set(state.frightened_state_time > 0)

        return obs


    @partial(jax.jit, static_argnums=(0,))
    def reset(self, key: PRNGKey, params: EnvParams) -> tuple[jnp.ndarray, State]:
        return Environment.reset(self, key, params)

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(self, key: chex.PRNGKey, params: EnvParams):
        state, _ = PacMan.reset(self, key)
        obs = self.get_obs(state)
        return obs, state

    @partial(jax.jit, static_argnums=(0,))
    def step(self,
             key: chex.PRNGKey,
             state: State,
             action: int,
             params: EnvParams):
        return Environment.step(self, key, state, action, params)

    @partial(jax.jit, static_argnums=(0,))
    def step_env(self,
                 key: chex.PRNGKey,
                 state: State,
                 action: int,
                 params: EnvParams):
        # Most of this was taken from
        # https://github.com/instadeepai/jumanji/blob/main/jumanji/environments/routing/pac_man/env.py

        # Collect updated state based on environment dynamics
        updated_state, collision_rewards = self._update_state(state, action)

        # Create next_state from updated state
        next_state = updated_state.replace(step_count=state.step_count + 1)  # type: ignore

        # Check if episode terminates
        num_pellets = next_state.pellets
        time_limit_exceeded = next_state.step_count >= self.time_limit
        all_pellets_found = num_pellets == 0
        dead = next_state.dead == 1
        done = time_limit_exceeded | dead | all_pellets_found

        reward = jnp.asarray(collision_rewards)
        obs = self.get_obs(next_state)
        return jax.lax.stop_gradient(obs), jax.lax.stop_gradient(next_state), reward, done, {}


if __name__ == "__main__":
    # jax.disable_jit(True)
    key = jax.random.PRNGKey(2024)

    env = PocMan()
    env_params = env.default_params

    reset_key, key = jax.random.split(key)
    obs, state = env.reset(key, env_params)

    step_key, key = jax.random.split(key)
    obs, state, rew, terminal, info = env.step(step_key, state, jnp.array(1), env_params)

    print()
