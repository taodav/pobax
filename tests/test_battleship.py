import jax
import jax.numpy as jnp
from jax import random

from pobax.envs.jax.battleship import Battleship, BattleShipState, place_ship_randomly


if __name__ == "__main__":
    # jax.disable_jit(True)
    board_5_4_placed = jnp.array([
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ])
    horz_3_allowed_5_4 = jnp.array([
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 0, 0, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
        [1, 1, 1, 1, 1, 1, 1, 1, 0, 0],
    ])
    vert_3_allowed_5_4 = jnp.array([
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 0, 0, 0, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 0, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    ])
    board = jnp.zeros((10, 10), dtype=int)
    lengths = (5, 4, 3, 2)
    samples = 1000

    key = random.PRNGKey(2025)

    # test place_samples
    for i in range(samples):
        place_key, key = random.split(key)
        new_board, new_pos, pose = place_ship_randomly(place_key, board_5_4_placed, 3)
        if pose.item() == 0:
            allowed = vert_3_allowed_5_4[new_pos]
        else:
            allowed = horz_3_allowed_5_4[new_pos]
        assert bool(allowed)

    env = Battleship()
    env_params = env.default_params

    # Now we test our step
    board = jnp.array([
        [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ], dtype=int)

    def ravel_idx(y: int, x: int):
        return y * env.rows + x

    reset_key, key = random.split(key)
    obs, state = env.reset(reset_key, env_params)
    assert jnp.all(obs.action_mask == 1)

    state = BattleShipState(hits_misses=jnp.zeros_like(board),
                            board=board)
    # Test miss
    a = (4, 3)
    step_key, key = random.split(key)
    obs, state, rew, term, info = env.step(step_key, state, ravel_idx(*a), env_params)

    assert state.hits_misses[a] == 1
    assert obs.obs[0] == 0
    action_idx = a[0] * env.rows + a[1]

    # check action mask is correct
    assert obs.action_mask[action_idx] == 0
    assert jnp.all(obs.action_mask[:action_idx] == 1) and jnp.all(obs.action_mask[action_idx + 1:] == 1)

    # Test hit
    a = (2, 8)
    step_key, key = random.split(key)
    obs, state, rew, term, info = env.step(step_key, state, ravel_idx(*a), env_params)

    assert state.hits_misses[a] == 2
    assert obs.obs[0] == 1

    # check action mask is correct
    assert obs.action_mask[a[0] * env.rows + a[1]] == 0
    assert obs.action_mask.sum() == (env.rows * env.cols - 2)

    a = (6, 1)
    almost_all_hit = board * 2
    almost_all_hit = almost_all_hit.at[a].set(0)

    state = state.replace(hits_misses=almost_all_hit)

    step_key, key = random.split(key)
    obs, state, rew, term, info = env.step(step_key, state, ravel_idx(*a), env_params)

    assert term
    assert rew == 100

    print("All tests passed.")
