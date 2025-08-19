import jax

from pobax.envs.jax.simple_chain import FullyObservableSimpleChain
from pobax.envs.wrappers.gymnax import ActionRepeatWrapper

def test_action_repeat():
    n_repeats = 3
    key = jax.random.PRNGKey(2025)

    simple_chain = FullyObservableSimpleChain(n=5)
    env_params = simple_chain.default_params

    env = ActionRepeatWrapper(simple_chain, n_repeats=n_repeats)

    reset_key, key = jax.random.split(key)
    obs, state = env.reset(key, env_params)

    # Now if we take a step, it should take us to index n_repeats
    step_key, key = jax.random.split(key)
    obs, state, reward, done, info = env.step(step_key, state, 0, env_params)
    assert state.pos_idx == n_repeats
    assert obs[n_repeats] == 1 and obs.sum() == 1

    # Now if we take a step again, we should be at the terminal state.
    step_key, key = jax.random.split(key)
    obs, state, reward, done, info = env.step(step_key, state, 0, env_params)
    assert done
    assert reward == 1
    assert obs[0] == 1 and state.pos_idx == 0

    # Now we're going to test double dones
    env = ActionRepeatWrapper(simple_chain, n_repeats=n_repeats * 2)
    reset_key, key = jax.random.split(key)
    obs, state = env.reset(key, env_params)

    # Now if we take a step, it should take us to the first done
    step_key, key = jax.random.split(key)
    obs, state, reward, done, info = env.step(step_key, state, 0, env_params)
    assert done
    assert reward == 1
    assert obs[0] == 1 and state.pos_idx == 0


if __name__ == "__main__":
    test_action_repeat()
