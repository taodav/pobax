from functools import partial
from typing import Union, Callable

import chex
import jax
import jax.numpy as jnp
import navix as nx
from navix.entities import Wall, Player, Goal
from navix.environments import Timestep
from navix.environments import Environment as NavixEnvironment
from navix.grid import random_positions, random_directions
from navix.rendering.cache import RenderingCache
from navix.states import State

import numpy as np

nav_maze_random_goal_01 = """
#####################
#...#...#...........#
#.#.#.#####.#####.#.#
#.#.#.#...#.....#.#.#
#.#.#.#...#.....#.#.#
#.#...#...#.....#.#.#
#.#####...#.#...#.#.#
#.....#...#.#...#.#.#
#####.#.###########.#
#...................#
#####################
"""

nav_maze_random_goal_02 = """
###############################
#.#...............#.......#...#
#.#.###.###########.###.#.#.#.#
#...#...#.............#.#...#.#
#.###.###.#############.#####.#
#...#...#.#.........#...#.....#
###.###.#.#.........#.#####.#.#
#.....#...#.........#.#.....#.#
#.###.#####.........#.#.....###
#.#.....#...........#.#.....#.#
#.#.....#.#.........#.#.....#.#
#.......#.#...........#.....#.#
#.#.....#.#########...#.....#.#
#.#.....#.......#.....#.....#.#
#.#####.#######.#.###.#######.#
#...#.........#.#.#.....#.....#
###.###.#####.#.#.#######.#####
#.......#.......#.............#
###############################
"""

nav_maze_random_goal_03 = """
#########################################
#.......................................#
#.#################.###########.#######.#
#.#...#...........#.#...#.............#.#
#.###.#...........#.#.#.#.............#.#
#...#.............#.#.#.#.............#.#
###.#.#...........#.###.#.............#.#
#.#...#...........#...#.#.............#.#
#.###.#...........###.#.#.............#.#
#...#.#...........#...#.#.............#.#
#.###.#...........#.###.#.............#.#
#.....#...........#.....#.............#.#
#.#####################.###############.#
#.#.........#.........#.......#.........#
#.#.#.###############.#.#######.#.#######
#...#.#.............#.#.......#.#.#.....#
#####.#.............#.#.......#.###.###.#
#.#...#.............#.#.......#.....#...#
#.#.###.............#.#.......#.#####.#.#
#...#.#.............#.#.......#.#...#.#.#
#.###.#.............#.#.......#.#.#.#.#.#
#...#...............#.#.......#...#.#.#.#
###.#.###############.#.......#####.#.#.#
#...#...#.....#.....#.#.......#...#.#.#.#
#.###.#.###.#.###.#.#.#.#######.#.###.#.#
#.....#.....#.....#.#...........#.....#.#
#########################################
"""


class Maze(NavixEnvironment):
    spec: str = "nav_maze_random_goal_01"

    def _reset(self, key: chex.PRNGKey, cache: Union[RenderingCache, None] = None) -> Timestep:
        player_pos_key, player_dir_key, goal_key = jax.random.split(key, 3)

        ascii_str = globals()[self.spec]
        grid = nx.grid.from_ascii_map(ascii_str)

        # Get our wall positions
        wall_pos = jnp.stack(jnp.nonzero(grid == -1), axis=1)
        walls = Wall.create(position=wall_pos)

        # player position
        player_pos = random_positions(player_pos_key, grid)
        player_dir = random_directions(player_dir_key)

        avail_goal_grid = grid.at[player_pos].set(-1)
        goal_pos = random_positions(goal_key, avail_goal_grid)

        # spawn goal and player
        player = Player.create(
            position=player_pos, direction=player_dir, pocket=nx.components.EMPTY_POCKET_ID
        )
        goal = Goal.create(position=goal_pos, probability=jnp.asarray(1.0))
        entities = {
            "player": player[None],
            "goal": goal[None],
            "wall": walls
        }

        state = State(
            key=key,
            grid=grid,
            cache=cache or RenderingCache.init(grid),
            entities=entities,
        )
        return Timestep(
            t=jnp.asarray(0, dtype=jnp.int32),
            observation=self.observation_fn(state),
            action=jnp.asarray(-1, dtype=jnp.int32),
            reward=jnp.asarray(0.0, dtype=jnp.float32),
            step_type=jnp.asarray(0, dtype=jnp.int32),
            state=state,
        )


def add_channel_obs_fn(state: State, obs_fn: Callable):
    # Add a channel in the last dimension
    categorical_obs = obs_fn(state)
    return categorical_obs[..., None]


radius = nx.observations.RADIUS
categorical_obs_fn = partial(add_channel_obs_fn, obs_fn=nx.observations.categorical_first_person)
categorical_obs_space = nx.spaces.Discrete.create(n_elements=9, shape=(radius + 1, radius * 2 + 1, 1))


nx.register_env(
    "Navix-DMLab-Maze-01-v0",
    lambda *args, **kwargs: Maze.create(
        # observation_fn=kwargs.pop("observation_fn", nx.observations.categorical_first_person),
        observation_fn=kwargs.pop("observation_fn", categorical_obs_fn),
        observation_space=categorical_obs_space,
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=11,
        width=21,
        max_steps=1000,
        spec="nav_maze_random_goal_01",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-RGB-01-v0",
    lambda *args, **kwargs: Maze.create(
        observation_fn=kwargs.pop("observation_fn", nx.observations.rgb_first_person),
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=11,
        width=21,
        max_steps=1000,
        spec="nav_maze_random_goal_01",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-02-v0",
    lambda *args, **kwargs: Maze.create(
        # observation_fn=kwargs.pop("observation_fn", nx.observations.categorical_first_person),
        observation_fn=kwargs.pop("observation_fn", categorical_obs_fn),
        observation_space=categorical_obs_space,
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=19,
        width=31,
        max_steps=2000,
        spec="nav_maze_random_goal_02",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-RGB-02-v0",
    lambda *args, **kwargs: Maze.create(
        observation_fn=kwargs.pop("observation_fn", nx.observations.rgb_first_person),
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=19,
        width=31,
        max_steps=2000,
        spec="nav_maze_random_goal_02",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-03-v0",
    lambda *args, **kwargs: Maze.create(
        # observation_fn=kwargs.pop("observation_fn", nx.observations.categorical_first_person),
        observation_fn=kwargs.pop("observation_fn", categorical_obs_fn),
        observation_space=categorical_obs_space,
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=27,
        width=41,
        max_steps=4000,
        spec="nav_maze_random_goal_03",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-RGB-03-v0",
    lambda *args, **kwargs: Maze.create(
        observation_fn=kwargs.pop("observation_fn", nx.observations.rgb_first_person),
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=27,
        width=41,
        max_steps=4000,
        spec="nav_maze_random_goal_03",
        *args,
    ),
)

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    key = jax.random.PRNGKey(2024)
    env = nx.make("Navix-DMLab-Maze-RGB-03-v0")
    tstep = env.reset(key)

    plt.imshow(tstep.observation)
    plt.show()


    print()
