from pathlib import Path

import jax

from pobax.envs.jax.rocksample import RockSample
from definitions import ROOT_DIR

def test_rocksample():
    seed = 2020
    # env_name = 'rocksample_7_8'
    env_name = 'rocksample_11_11'

    key = jax.random.PRNGKey(seed)
    reset_key, env_key, key = jax.random.split(key, 3)

    config_path = Path(ROOT_DIR, 'pobax', 'envs', 'configs', f'{env_name}_config.json')
    env = RockSample(env_key, config_path=config_path)

    env_params = env.default_params
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

    # now we try sampling
    step_key, key = jax.random.split(key)
    obs, state, rew, _, _ = env.step(step_key, state, 4, env_params)
    assert obs[2 * env.size + rock_to_check - 1] == 0
    assert rew > 0

    # now we check that it's turned into a bad rock
    actions_to_rock = [0, 2]  # for 11, 11
    rew = 0
    for a in actions_to_rock:
        step_key, key = jax.random.split(key)
        obs, state, rew, terminal, info = env.step(step_key, state, a, env_params)
        assert obs[2 * env.size + rock_to_check - 1] == 0.

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

    # now we check checking bad rock.
    rock_to_check = 5
    step_key, key = jax.random.split(key)
    obs, state, _, _, _ = env.step(step_key, state, 4 + rock_to_check, env_params)

    assert obs[2 * env.size + rock_to_check - 1] == -1

    # now we try sampling
    step_key, key = jax.random.split(key)
    obs, state, rew, _, _ = env.step(step_key, state, 4, env_params)
    assert obs[2 * env.size + rock_to_check - 1] == 0
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
