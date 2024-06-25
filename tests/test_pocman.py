"""
We test PocMan here. Since PacMan is (seemingly) tested in Jumanji,
we just need to test our observation and line-of-sight map.
"""
import jax
import jax.numpy as jnp
import matplotlib

# matplotlib.use('TkAgg')
matplotlib.use('Qt5Agg')

from pobax.envs.pocman import PocMan


def unpack_obs(obs: jnp.ndarray):
    walls = obs[:4]
    smell_pellet = obs[4]
    hear_ghost = obs[5]
    ghost_in_los = obs[6:10]
    powerpill = obs[10]
    return walls, smell_pellet, hear_ghost, ghost_in_los, powerpill


if __name__ == "__main__":
    # jax.disable_jit(True)

    seed = 2024
    key = jax.random.PRNGKey(seed)
    reset_key, key = jax.random.split(key)

    env = PocMan()
    env_params = env.default_params

    obs, state = env.reset(reset_key, env_params)

    def get_state_los_map(state):
        pos_x, pos_y = state.player_locations.x, state.player_locations.y
        if isinstance(pos_x, jnp.ndarray):
            pos_x, pos_y = pos_x.item(), pos_y.item()
        los_map = env.line_sight_map[pos_x, pos_y]
        return los_map
    los_map = get_state_los_map(state)
    los_map_no_agent = los_map.at[:, state.player_locations.x, state.player_locations.y].set(0)

    assert jnp.all(los_map_no_agent[0] == 0)

    right_view = los_map[3, state.player_locations.x, state.player_locations.y + 1:]
    assert jnp.all(right_view == jnp.array([1, 1, 1, 1, 1, 0, 0, 0, 0]))
    assert jnp.all(los_map.at[3, state.player_locations.x, state.player_locations.y:].set(0)[3] == 0)

    left_view = los_map[1, state.player_locations.x, :state.player_locations.y]
    assert jnp.all(left_view == jnp.array([0, 0, 0, 0, 1, 1, 1, 1, 1]))
    assert jnp.all(los_map.at[1, state.player_locations.x, :state.player_locations.y + 1].set(0)[1] == 0)

    step_key, key = jax.random.split(key)
    obs, state, rew, terminal, info = env.step(step_key, state, 1, env_params)

    step_key, key = jax.random.split(key)
    obs, state, rew, terminal, info = env.step(step_key, state, 0, env_params)

    los_map = get_state_los_map(state)
    up_view = los_map[0, :state.player_locations.x, state.player_locations.y]
    down_view = los_map[2, state.player_locations.x + 1:, state.player_locations.y]
    assert up_view[-1] == 1 and jnp.all(up_view[:-1] == 0)
    assert down_view[0] == 1 and jnp.all(down_view[1:] == 0)

    step_key, key = jax.random.split(key)
    obs, state, rew, terminal, info = env.step(step_key, state, 0, env_params)

    step_key, key = jax.random.split(key)
    obs, state, rew, terminal, info = env.step(step_key, state, 1, env_params)

    step_key, key = jax.random.split(key)
    obs, state, rew, terminal, info = env.step(step_key, state, 1, env_params)

    # go up the corridor to greet red/pink guy
    for i in range(4):
        step_key, key = jax.random.split(key)
        obs, state, rew, terminal, info = env.step(step_key, state, 0, env_params)

    # There should be 1 pellet dist 1 away.
    assert obs[4] == 1

    # LOS should be all zeros
    assert jnp.all(obs[6:10] == jnp.array([0, 0, 0, 0]))

    step_key, key = jax.random.split(key)
    obs, state, rew, terminal, info = env.step(step_key, state, 2, env_params)

    step_key, key = jax.random.split(key)
    obs, state, rew, terminal, info = env.step(step_key, state, 4, env_params)

    assert rew == 0

    # walls
    assert obs[0] == 1 and obs[2] == 1
    assert obs[1] == 0 and obs[3] == 0

    # There should be no pellets dist 1 away.
    assert obs[4] == 0

    # There's a ghost mahattan distance 3 away, so we should be able to hear it
    assert obs[5] == 1

    # Finally, LOS should be 1, 0, 0, 0
    assert jnp.all(obs[6:10] == jnp.array([1, 0, 0, 0]))

    # Now let's go get a power up
    for i in range(3):
        step_key, key = jax.random.split(key)
        obs, state, rew, terminal, info = env.step(step_key, state, 2, env_params)

    for i in range(5):
        step_key, key = jax.random.split(key)
        obs, state, rew, terminal, info = env.step(step_key, state, 1, env_params)

    step_key, key = jax.random.split(key)
    obs, state, rew, terminal, info = env.step(step_key, state, 2, env_params)

    for i in range(4):
        step_key, key = jax.random.split(key)
        obs, state, rew, terminal, info = env.step(step_key, state, 4, env_params)

    # Collect the power up
    step_key, key = jax.random.split(key)
    obs, state, rew, terminal, info = env.step(step_key, state, 2, env_params)
    assert rew > 0
    assert obs[-1] == 1

    for i in range(2):
        step_key, key = jax.random.split(key)
        obs, state, rew, terminal, info = env.step(step_key, state, 0, env_params)

    for i in range(5):
        step_key, key = jax.random.split(key)
        obs, state, rew, terminal, info = env.step(step_key, state, 3, env_params)

    for i in range(2):
        step_key, key = jax.random.split(key)
        obs, state, rew, terminal, info = env.step(step_key, state, 0, env_params)

    # Here we've just eaten a ghost
    assert rew > 0

    print('Tests all passed.')
