import jax
import jax.numpy as jnp

from pobax.envs import get_env
from pobax.utils.bfs_grid_solver import jax_shortest_path_policy

if __name__ == "__main__":
    env_name = 'Navix-DMLab-Maze-02-v0'
    seed = 2025
    n_envs = 100

    key = jax.random.PRNGKey(seed)
    env_key, key = jax.random.split(key)

    vmap_shortest_path = jax.vmap(jax_shortest_path_policy, in_axes=[None, 0, 0, 0])

    env, env_params = get_env(env_name, env_key, gamma=0.99)
    reset_rng = jax.random.split(key, n_envs)
    obsv, env_state = env.reset(reset_rng, env_params)

    state = env_state.env_state.state
    grid = state.grid[0] * (-1)
    players = state.entities['player']
    start_pos = players.position.squeeze()
    start_orientation = players.direction.squeeze()
    goal_pos = state.entities['goal'].position.squeeze()

    policies_fixed, counts = vmap_shortest_path(grid, start_pos, start_orientation, goal_pos)

    disc_returns = env.gamma ** counts
    avg_disc_returns = jnp.mean(disc_returns)

    print()
