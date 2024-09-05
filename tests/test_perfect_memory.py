from pathlib import Path

import jax
import jax.numpy as jnp

from pobax.envs.battleship import Battleship, BattleShipState
from pobax.envs.battleship import PerfectMemoryWrapper as BSPerfectMemoryWrapper
from pobax.envs.jax.rocksample import RockSample
from pobax.envs.jax.rocksample import PerfectMemoryWrapper as RSPerfectMemoryWrapper
from definitions import ROOT_DIR


def test_battleship():
    key = jax.random.PRNGKey(2025)

    env = BSPerfectMemoryWrapper(Battleship())
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

    state = BattleShipState(hits_misses=jnp.zeros_like(board),
                            board=board)
    # Test miss
    a = (4, 3)
    step_key, key = jax.random.split(key)
    obs, state, rew, term, info = env.step(step_key, state, ravel_idx(*a), env_params)

    assert state.hits_misses[a] == 1
    assert obs[a] == -1

    # Test hit
    prev_a = a
    a = (2, 8)
    step_key, key = jax.random.split(key)
    obs, state, rew, term, info = env.step(step_key, state, ravel_idx(*a), env_params)

    assert state.hits_misses[a] == 2
    assert obs[prev_a] == -1
    assert obs[a] == 1

    a = (6, 1)
    almost_all_hit = board * 2
    almost_all_hit = almost_all_hit.at[a].set(0)

    state = state.replace(hits_misses=almost_all_hit)

    step_key, key = jax.random.split(key)
    obs, state, rew, term, info = env.step(step_key, state, ravel_idx(*a), env_params)

    assert term
    assert rew == 100

    print("All tests passed.")

def test_rocksample():
    seed = 2020
    # env_name = 'rocksample_7_8'
    env_name = 'rocksample_11_11'

    key = jax.random.PRNGKey(seed)
    reset_key, env_key, key = jax.random.split(key, 3)

    config_path = Path(ROOT_DIR, 'pobax', 'envs', 'configs', f'{env_name}_config.json')
    env = RockSample(env_key, config_path=config_path)
    env_params = env.default_params
    env = RSPerfectMemoryWrapper(env)
    obs, state = env.reset(reset_key, env_params)

    # Basic transition
    # actions_to_rock = [0, 0, 0, 0]  # for 7, 8
    actions_to_rock = [0, 1, 1]  # for 11, 11
    rew = 0
    for a in actions_to_rock:
        step_key, key = jax.random.split(key)
        obs, state, rew, terminal, info = env.step(step_key, state, a, env_params)

    # now we check checking
    rock_to_check = 3
    step_key, key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, 4 + rock_to_check, env_params)
    assert obs[2 * env.size + rock_to_check - 1] == 1.

    # now we check that the rock observation stays
    actions_to_rock = [0, 2]  # for 11, 11
    rew = 0
    for a in actions_to_rock:
        step_key, key = jax.random.split(key)
        obs, state, rew, terminal, info = env.step(step_key, state, a, env_params)
        assert obs[2 * env.size + rock_to_check - 1] == 1.

    # now we try sampling
    step_key, key = jax.random.split(key)
    obs, state, rew, _, _ = env.step(step_key, state, 4, env_params)
    assert obs[2 * env.size + rock_to_check - 1] == -1.
    assert rew > 0

    # now we check that it's turned into a bad rock
    actions_to_rock = [0, 2]  # for 11, 11
    rew = 0
    for a in actions_to_rock:
        step_key, key = jax.random.split(key)
        obs, state, rew, terminal, info = env.step(step_key, state, a, env_params)
        assert obs[2 * env.size + rock_to_check - 1] == -1.

    # now we check checking a previous good, currently bad rock.
    rock_to_check = 3
    step_key, key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, 4 + rock_to_check, env_params)

    assert obs[2 * env.size + rock_to_check - 1] == -1


    # Basic transition to bad rock
    actions_to_rock = [1, 2]  # for 11, 11
    rew = 0
    for a in actions_to_rock:
        step_key, key = jax.random.split(key)
        obs, state, rew, terminal, info = env.step(step_key, state, a, env_params)

    # now we check checking a bad rock.
    rock_to_check = 5
    step_key, key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, 4 + rock_to_check, env_params)

    assert obs[2 * env.size + rock_to_check - 1] == -1

    # now we try sampling
    step_key, key = jax.random.split(key)
    obs, state, rew, _, _ = env.step(step_key, state, 4, env_params)
    assert obs[2 * env.size + rock_to_check - 1] == -1.
    assert rew < 0

    # Now we see if terminal is behaving correctly
    actions_to_terminal = [1, 1, 1, 1]
    rew, term = 0, False
    for a in actions_to_terminal:
        step_key, key = jax.random.split(key)
        obs, state, rew, term, info = env.step(step_key, state, a, env_params)

    assert term and rew > 0

    print("All tests pass.")


if __name__ == "__main__":
    jax.disable_jit(True)

    test_rocksample()
    # test_battleship()

    # env_params = RockSample.default_params
    # rock_positions = RockSample.generate_map(env_params.size, env_params.k, jax.random.PRNGKey(1))
    # env = RockSample(rock_positions)
    #
    # key = jax.random.PRNGKey(2)
    # reset_key, key = jax.random.split(key)
    #
    # obs, state = env.reset(reset_key, env_params)
    #
    # for i in range(10000):
    #     act_key, step_key, key = jax.random.split(key, 3)
    #     action = env.action_space(env_params).sample(act_key)
    #     next_obs, next_state, reward, terminal, info = env.step(step_key, state, action, env_params)
    #     if terminal:
    #         reset_key, key = jax.random.split(key)
    #         obs, state = env.reset(reset_key, env_params)
    #         if i < 9999:
    #             print()

    print()
