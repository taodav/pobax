from functools import partial
from typing import Tuple, Optional, Union

import jax
import jax.numpy as jnp
import chex
from gymnax.environments import environment, spaces
import navix as nx
from navix.environments import Environment as NavixEnvironment
import numpy as np

from pobax.envs.wrappers.gymnax import GymnaxWrapper
from pobax.utils.grid import precompute_rays

class NavixGymnaxWrapper:
    def __init__(self, env: NavixEnvironment):
        self._env = env
        self.gamma = 0.99

    @property
    def default_params(self):
        return environment.EnvParams(max_steps_in_episode=self._env.max_steps)

    def observation_space(self, params):
        return spaces.Box(
            low=self._env.observation_space.minimum,
            high=self._env.observation_space.maximum,
            shape=self._env.observation_space.shape,
            dtype=self._env.observation_space.dtype,
        )

    def action_space(self, params):
        # return spaces.Discrete(
        #     num_categories=self._env.action_space.maximum.item() + 1,
        # )
        return spaces.Discrete(3)

    @partial(jax.jit, static_argnums=(0, -1))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, nx.environments.Timestep]:
        timestep = self._env.reset(key)

        return timestep.observation, timestep

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
            self,
            key: chex.PRNGKey,
            state: nx.environments.Timestep,
            action: int,
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, nx.environments.Timestep, float, bool, dict]:
        timestep = self._env.step(state, action)
        return timestep.observation, timestep, timestep.reward, timestep.is_done(), {}

# class

class MazeFoVWrapper(GymnaxWrapper):
    def __init__(self, env: NavixGymnaxWrapper,
                 mask_value: int = -1):
        super().__init__(env)
        radius = nx.observations.RADIUS

        self.grid_rows, self.grid_cols, _ = env.observation_space(env.default_params).shape
        self.agent_r, self.agent_c = radius, radius

        # Precompute these constants
        # Agent position is always (radius, radius)
        self.rays, self.ray_lengths = precompute_rays(self.grid_rows, self.grid_cols, self.agent_r, self.agent_c)
        self.mask_value = mask_value

    # -------------------------------
    # Field-of-view (FOV) computation
    # -------------------------------

    def is_visible(self, idx, obs):
        """
        Given a target cell index (as a 2-element array [r, c]) and the current observation,
        return True if that cell is visible to the agent.

        The logic is:
          - If the cell is the agent’s own cell, it is visible.
          - Otherwise, we “walk” the ray from the agent to the cell.
            If any intermediate cell (i.e. before the target) is a wall, then the view is blocked.
        """
        r, c = idx[0], idx[1]
        WALL = 1

        # For the agent’s own cell, always return True.
        def check_ray(_):
            length = self.ray_lengths[r, c]

            # Loop over the cells along the ray (except the final one)
            def body(i, vis):
                cell = self.rays[r, c, i]  # cell is a 2-element vector [row, col]
                # If this intermediate cell is a wall, then block the view.
                return jax.lax.cond(
                    (i < length - 1) & (obs[cell[0], cell[1]] == WALL),
                    lambda _: False,
                    lambda _: vis,
                    operand=None,
                )

            return jax.lax.fori_loop(0, length - 1, body, True)

        return jax.lax.cond((r == self.agent_r) & (c == self.agent_c),
                            lambda _: True,
                            check_ray,
                            operand=None)


    def compute_visibility(self, obs):
        """
        Compute a boolean (GRID_ROWS x GRID_COLS) visibility mask given the observation.
        """
        # Create an array of all grid cell indices.
        indices = jnp.array([[r, c] for r in range(self.grid_rows) for c in range(self.grid_cols)], dtype=jnp.int32)
        vis_flat = jax.vmap(lambda idx: self.is_visible(idx, obs))(indices)
        return vis_flat.reshape((self.grid_rows, self.grid_cols))

    # -------------------------------
    # The jitted masking function
    # -------------------------------

    @partial(jax.jit, static_argnums=0)
    def mask_observation(self, obs):
        """
        Given an observation (a 4×7x2 array) returns a new array where cells that are
        not in the agent's field-of-view are replaced by mask_value.

        Parameters:
          obs: jnp.ndarray of shape (4,7,2), where obs[..., 0] is the walls map.
          mask_value: the value to fill in for unseen cells (default: -1)

        Returns:
          A 4×7x2 jnp.ndarray where unseen cells are masked out for both channels.
        """
        grid_obs = obs[..., 0]
        vis = self.compute_visibility(grid_obs)
        vmapped_where = jax.vmap(jnp.where, in_axes=[None, -1, None], out_axes=-1)
        return vmapped_where(vis, obs, self.mask_value)

    @partial(jax.jit, static_argnums=(0,-1))
    def reset(
            self, key: chex.PRNGKey, params: Optional[environment.EnvParams] = None
    ) -> Tuple[chex.Array, environment.EnvState]:
        obs, state = self._env.reset(key, params)
        return self.mask_observation(obs), state

    @partial(jax.jit, static_argnums=(0, -1))
    def step(
            self,
            key: chex.PRNGKey,
            state: environment.EnvState,
            action: Union[int, float],
            params: Optional[environment.EnvParams] = None,
    ) -> Tuple[chex.Array, environment.EnvState, float, bool, dict]:
        obs, state, reward, done, info = self._env.step(key, state, action, params)
        return self.mask_observation(obs), state, reward, done, info


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
