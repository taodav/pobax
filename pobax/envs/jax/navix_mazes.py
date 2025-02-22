from functools import partial
from typing import Union, Callable, Dict, Any, Tuple

import chex
from flax import struct
import jax
import jax.numpy as jnp
from jax import Array
import navix as nx
from navix import rewards, terminations, observations, transitions
from navix.actions import DEFAULT_ACTION_SET
from navix.entities import Wall, Player, Goal
from navix.environments import Timestep
from navix.environments import Environment as NavixEnvironment
from navix.grid import random_positions, random_directions
from navix.rendering.cache import RenderingCache
from navix.states import State
from navix.spaces import Space, Discrete

from pobax.utils.grid import agent_centric_map, agent_position_map
import numpy as np

# nav_maze_random_goal_00 = """
# ##########
# #...#....#
# #...#....#
# #...####.#
# ###......#
# #.####...#
# #........#
# ##########
# """
nav_maze_random_goal_00 = """
##############
#...#...#....#
#.#.#.#####.##
#.#.#....#...#
#.#.#.#..#...#
#.#...#..#.#.#
#.#####..#.#.#
#.....#....#.#
##############
"""

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

nav_random_goal_01 = """
#####################
#...................#
#.........#.........#
#.........#.........#
#.........#....#....#
#.........#....#....#
#.........#....#....#
#.........#....#....#
#.........#....#....#
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

class ASCIIMaze(NavixEnvironment):
    grid: jnp.ndarray = struct.field(pytree_node=False, default=None)
    wall_pos: jnp.ndarray = struct.field(pytree_node=False, default=None)

    @classmethod
    def create(
            cls,
            height: int,
            width: int,
            max_steps: int | None = None,
            observation_fn: Callable[[State], Array] = observations.symbolic,
            reward_fn: Callable[[State, Array, State], Array] = rewards.DEFAULT_TASK,
            termination_fn: Callable[
                [State, Array, State], Array
            ] = terminations.DEFAULT_TERMINATION,
            transitions_fn: Callable[
                [State, Array, Tuple[Callable[[State], State], ...]], State
            ] = transitions.DEFAULT_TRANSITION,
            action_set: Tuple[Callable[[State], State], ...] = DEFAULT_ACTION_SET,
            observation_space: Space | None = None,
            action_space: Space | None = None,
            reward_space: Space | None = None,
            spec: str = 'nav_maze_random_goal_01',
            **kwargs,
    ) -> NavixEnvironment:

        if observation_space is None:
            observation_space = cls._get_obs_space_from_fn(
                width, height, observation_fn
            )
        ascii_str = globals()[spec]

        grid = nx.grid.from_ascii_map(ascii_str)

        # Get our wall positions
        wall_pos = jnp.stack(jnp.nonzero(grid == -1), axis=1)
        return cls(
            height=height,
            width=width,
            max_steps=max_steps,
            observation_fn=observation_fn,
            reward_fn=reward_fn,
            termination_fn=termination_fn,
            transitions_fn=transitions_fn,
            action_set=action_set,
            observation_space=observation_space,
            action_space=action_space,
            reward_space=reward_space,
            grid=grid,
            wall_pos=wall_pos,
            **kwargs,
        )

    def _reset(self, key: chex.PRNGKey, cache: Union[RenderingCache, None] = None) -> Timestep:
        walls = Wall.create(position=self.wall_pos)
        player_pos_key, player_dir_key, goal_key = jax.random.split(key, 3)

        # player position
        player_pos = random_positions(player_pos_key, self.grid)
        player_dir = random_directions(player_dir_key)

        avail_goal_grid = self.grid.at[player_pos].set(-1)
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
            grid=self.grid,
            cache=cache or RenderingCache.init(self.grid),
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


def categorical_one_hot_first_person(state: State):
    # Add a channel in the last dimension
    categorical_obs = nx.observations.categorical_first_person(state)
    wall_tag, goal_tag = state.entities['wall'].tag[0], state.entities['goal'].tag[0]
    one_hot_wall = (categorical_obs == wall_tag).astype(int)
    one_hot_goal = (categorical_obs == goal_tag).astype(int)

    return jnp.stack((one_hot_wall, one_hot_goal), axis=-1)

def categorical_full_first_person(state: State):
    wall_map = state.grid
    goal_map = jnp.zeros_like(wall_map).at[state.entities['goal'].position].set(1)
    wall_and_goal = jnp.stack((wall_map, goal_map), axis=-1)

    player = state.get_player()
    ac_wall_and_goal = agent_centric_map(wall_and_goal, player.position, player.direction)
    return ac_wall_and_goal

def categorical_full_position_encoded(state: State):
    wall_map = state.grid
    goal_map = jnp.zeros_like(wall_map).at[state.entities['goal'].position].set(1)
    wall_and_goal = jnp.stack((wall_map, goal_map), axis=-1)

    player = state.get_player()
    ap_wall_and_goal = agent_position_map(wall_and_goal, player.position)

    # now we do one-hot directional encoding
    direction_vec = jnp.eye(4)[player.direction][None, None, ...]
    h, w = ap_wall_and_goal.shape[:2]
    one_hot_direction = direction_vec.repeat(h, axis=0).repeat(w, axis=1)

    obs = jnp.concatenate((ap_wall_and_goal, one_hot_direction), axis=-1)
    return obs

# def categorical_one_hot(state: State):
#     # Add a channel in the last dimension
#     categorical_obs = nx.observation
#     s.categorical(state)
#     player = state.entities['player']
#     player_tag, goal_tag, wall_tag = player.tag[0], state.entities['goal'].tag[0], state.entities['wall'].tag[0]
#     direction_vec = jnp.eye(4)[player.direction][None, ...]
#     one_hot_player = (categorical_obs == player_tag).astype(int)
#     # TODO: encode direction
#     one_hot_wall = (categorical_obs == wall_tag).astype(int)
#     one_hot_goal = (categorical_obs == goal_tag).astype(int)
#
#     h, w = categorical_obs.shape
#     one_hot_direction = direction_vec.repeat(h, axis=0).repeat(w, axis=1)
#
#     return jnp.concatenate((one_hot_player[..., None], one_hot_goal[..., None], one_hot_wall[..., None],
#                             one_hot_direction), axis=-1)
#
# def categorical_many_hot(state: State):
#     grid = state.grid
#     h, w = grid.shape
#     player = state.entities['player']
#     goal = state.entities['goal']
#     agent_r, agent_c = jnp.squeeze(player.position)
#     goal_r, goal_c = jnp.squeeze(goal.position)
#     player_r_one_hot = jnp.zeros(h).at[agent_r].set(1)
#     player_c_one_hot = jnp.zeros(w).at[agent_c].set(1)
#     goal_r_one_hot = jnp.zeros(h).at[goal_r].set(1)
#     goal_c_one_hot = jnp.zeros(w).at[goal_c].set(1)
#     dir_one_hot = jnp.zeros(4).at[jnp.squeeze(player.direction)].set(1)
#     return jnp.concatenate([player_r_one_hot, player_c_one_hot,
#                             goal_r_one_hot, goal_c_one_hot,
#                             dir_one_hot])

radius = nx.observations.RADIUS
categorical_first_person_obs_space = nx.spaces.Discrete.create(n_elements=9, shape=(radius + 1, radius * 2 + 1, 2))

def categorical_full_obs_space_fn(h, w):
    one_side = 2 * max(h, w) - 1
    return nx.spaces.Discrete.create(n_elements=9, shape=(one_side, one_side, 2))

def categorical_full_positional_obs_space_fn(h, w):
    return nx.spaces.Discrete.create(n_elements=9, shape=(2 * h - 1, 2 * w - 1, 2 + 4))
categorical_full_positional_obs_space = nx.spaces.Discrete.create(n_elements=9, shape=(radius + 1, radius * 2 + 1, 2))

nx.register_env(
    "Navix-DMLab-Maze-00-v0",
    lambda *args, **kwargs: ASCIIMaze.create(
        # observation_fn=kwargs.pop("observation_fn", nx.observations.categorical_first_person),
        observation_fn=kwargs.pop("observation_fn", categorical_one_hot_first_person),
        observation_space=categorical_first_person_obs_space,
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=9,
        width=14,
        max_steps=1000,
        spec="nav_maze_random_goal_00",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-F-00-v0",
    lambda *args, **kwargs: ASCIIMaze.create(
        # observation_fn=kwargs.pop("observation_fn", nx.observations.categorical_first_person),
        # observation_fn=kwargs.pop("observation_fn", observations.symbolic),
        # observation_space=categorical_obs_space_fn(9, 14),

        # observation_fn=kwargs.pop("observation_fn", categorical_full_first_person),
        # observation_space=categorical_full_obs_space_fn(9, 14),

        observation_fn=kwargs.pop("observation_fn", categorical_full_position_encoded),
        observation_space=categorical_full_positional_obs_space_fn(9, 14),
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=9,
        width=14,
        max_steps=1000,
        spec="nav_maze_random_goal_00",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-RGB-00-v0",
    lambda *args, **kwargs: ASCIIMaze.create(
        observation_fn=kwargs.pop("observation_fn", nx.observations.rgb_first_person),
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=9,
        width=14,
        max_steps=1000,
        spec="nav_maze_random_goal_00",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-01-v0",
    lambda *args, **kwargs: ASCIIMaze.create(
        # observation_fn=kwargs.pop("observation_fn", nx.observations.categorical_first_person),
        observation_fn=kwargs.pop("observation_fn", categorical_one_hot_first_person),
        observation_space=categorical_first_person_obs_space,
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=11,
        width=21,
        max_steps=2000,
        spec="nav_maze_random_goal_01",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-F-01-v0",
    lambda *args, **kwargs: ASCIIMaze.create(
        # observation_fn=kwargs.pop("observation_fn", nx.observations.categorical_first_person),
        # observation_fn=kwargs.pop("observation_fn", categorical_full_first_person),
        # observation_space=categorical_full_obs_space_fn(11, 21),

        observation_fn=kwargs.pop("observation_fn", categorical_full_position_encoded),
        observation_space=categorical_full_positional_obs_space_fn(11, 21),
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=11,
        width=21,
        max_steps=2000,
        spec="nav_maze_random_goal_01",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-TestMaze-F-01-v0",
    lambda *args, **kwargs: ASCIIMaze.create(
        # observation_fn=kwargs.pop("observation_fn", nx.observations.categorical_first_person),
        observation_fn=kwargs.pop("observation_fn", categorical_full_first_person),
        observation_space=categorical_full_obs_space_fn(11, 21),
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=11,
        width=21,
        max_steps=2000,
        spec="nav_random_goal_01",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-RGB-01-v0",
    lambda *args, **kwargs: ASCIIMaze.create(
        observation_fn=kwargs.pop("observation_fn", nx.observations.rgb_first_person),
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=11,
        width=21,
        max_steps=2000,
        spec="nav_maze_random_goal_01",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-02-v0",
    lambda *args, **kwargs: ASCIIMaze.create(
        # observation_fn=kwargs.pop("observation_fn", nx.observations.categorical_first_person),
        observation_fn=kwargs.pop("observation_fn", categorical_one_hot_first_person),
        observation_space=categorical_first_person_obs_space,
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=19,
        width=31,
        max_steps=4000,
        spec="nav_maze_random_goal_02",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-F-02-v0",
    lambda *args, **kwargs: ASCIIMaze.create(
        # observation_fn=kwargs.pop("observation_fn", nx.observations.categorical_first_person),
        # observation_fn=kwargs.pop("observation_fn", categorical_full_first_person),
        # observation_space=categorical_full_obs_space_fn(19, 31),

        observation_fn=kwargs.pop("observation_fn", categorical_full_position_encoded),
        observation_space=categorical_full_positional_obs_space_fn(19, 31),
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=19,
        width=31,
        max_steps=4000,
        spec="nav_maze_random_goal_02",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-RGB-02-v0",
    lambda *args, **kwargs: ASCIIMaze.create(
        observation_fn=kwargs.pop("observation_fn", nx.observations.rgb_first_person),
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=19,
        width=31,
        max_steps=4000,
        spec="nav_maze_random_goal_02",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-03-v0",
    lambda *args, **kwargs: ASCIIMaze.create(
        # observation_fn=kwargs.pop("observation_fn", nx.observations.categorical_first_person),
        observation_fn=kwargs.pop("observation_fn", categorical_one_hot_first_person),
        observation_space=categorical_first_person_obs_space,
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=27,
        width=41,
        max_steps=6000,
        spec="nav_maze_random_goal_03",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-F-03-v0",
    lambda *args, **kwargs: ASCIIMaze.create(
        # observation_fn=kwargs.pop("observation_fn", nx.observations.categorical_first_person),
        # observation_fn=kwargs.pop("observation_fn", categorical_full_first_person),
        # observation_space=categorical_full_obs_space_fn(27, 41),

        observation_fn=kwargs.pop("observation_fn", categorical_full_position_encoded),
        observation_space=categorical_full_positional_obs_space_fn(27, 41),
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=27,
        width=41,
        max_steps=6000,
        spec="nav_maze_random_goal_03",
        *args,
    ),
)

nx.register_env(
    "Navix-DMLab-Maze-RGB-03-v0",
    lambda *args, **kwargs: ASCIIMaze.create(
        observation_fn=kwargs.pop("observation_fn", nx.observations.rgb_first_person),
        reward_fn=kwargs.pop("reward_fn", nx.rewards.on_goal_reached),
        termination_fn=kwargs.pop("termination_fn", nx.terminations.on_goal_reached),
        height=27,
        width=41,
        max_steps=6000,
        spec="nav_maze_random_goal_03",
        *args,
    ),
)

