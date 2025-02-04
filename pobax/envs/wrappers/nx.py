from functools import partial
from typing import Tuple, Optional, Union

import jax
import jax.numpy as jnp
import chex
from flax import struct
from gymnax.environments import environment, spaces
import navix as nx
from navix.environments import Environment as NavixEnvironment

from pobax.envs.wrappers.gymnax import GymnaxWrapper
from pobax.utils.grid import precompute_rays

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

# class

class MazeFoVWrapper(GymnaxWrapper):
    def __init__(self, env: NavixEnvironment):
        super().__init__(env)

        self.rays, self.ray_lengths = precompute_rays()

# Precompute these constants (they will be embedded into the jitted code)
# RAYS, RAY_LENGTHS = precompute_rays()


# -------------------------------
# Field-of-view (FOV) computation
# -------------------------------

def is_visible(idx, obs):
    """
    Given a target cell index (as a 2-element array [r, c]) and the current observation,
    return True if that cell is visible to the agent.

    The logic is:
      - If the cell is the agent’s own cell, it is visible.
      - Otherwise, we “walk” the ray from the agent to the cell.
        If any intermediate cell (i.e. before the target) is a wall, then the view is blocked.
    """
    r, c = idx[0], idx[1]

    # For the agent’s own cell, always return True.
    def check_ray(_):
        length = RAY_LENGTHS[r, c]

        # Loop over the cells along the ray (except the final one)
        def body(i, vis):
            cell = RAYS[r, c, i]  # cell is a 2-element vector [row, col]
            # If this intermediate cell is a wall, then block the view.
            return jax.lax.cond(
                (i < length - 1) & (obs[cell[0], cell[1]] == WALL),
                lambda _: False,
                lambda _: vis,
                operand=None,
            )

        return jax.lax.fori_loop(0, length - 1, body, True)

    return jax.lax.cond((r == AGENT_R) & (c == AGENT_C),
                        lambda _: True,
                        check_ray,
                        operand=None)


def compute_visibility(obs):
    """
    Compute a boolean (GRID_ROWS x GRID_COLS) visibility mask given the observation.
    """
    # Create an array of all grid cell indices.
    indices = jnp.array([[r, c] for r in range(GRID_ROWS) for c in range(GRID_COLS)], dtype=jnp.int32)
    vis_flat = jax.vmap(lambda idx: is_visible(idx, obs))(indices)
    return vis_flat.reshape((GRID_ROWS, GRID_COLS))


# -------------------------------
# The jitted masking function
# -------------------------------

@jax.jit
def mask_observation(obs, mask_value=-1):
    """
    Given an observation (a 4×7 array) returns a new array where cells that are
    not in the agent's field-of-view are replaced by mask_value.

    Parameters:
      obs: jnp.ndarray of shape (4,7)
      mask_value: the value to fill in for unseen cells (default: -1)

    Returns:
      A 4×7 jnp.ndarray where unseen cells are masked out.
    """
    vis = compute_visibility(obs)
    return jnp.where(vis, obs, mask_value)


# -------------------------------
# Example usage
# -------------------------------
if __name__ == '__main__':
    # Suppose we have an observation where 1 = wall and 0 = free space.
    # (Here we set up an arbitrary example.)
    obs = jnp.array([
        [0, 0, 0, 0, 0, 0, 1],
        [0, 1, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
    ])
    masked = mask_observation(obs)
    print("Original observation:")
    print(obs)
    print("Masked observation:")
    print(masked)
