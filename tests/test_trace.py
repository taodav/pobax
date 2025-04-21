from argparse import Action

import jax
import numpy as np

from pobax.envs.jax.simple_chain import FullyObservableSimpleChain, FullyObservableImageSimpleChain
from pobax.envs.wrappers.gymnax import ActionConcatWrapper
from pobax.envs.wrappers.trace import TraceFeaturesWrapper


def test_simple_chain():
    n = 10
    rng = jax.random.PRNGKey(2025)

    unwrapped_env = FullyObservableSimpleChain(n=n)
    env_params = unwrapped_env.default_params

    env = ActionConcatWrapper(unwrapped_env)
    env = TraceFeaturesWrapper(env, trace_in_obs=True)

    n_lambdas = len(env.lambdas)

    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)
    obses = [obs]
    for i in range(n):
        rng, step_rng, act_rng = jax.random.split(rng, 3)
        obs, state, r, done, _ = env.step(step_rng, state, env.action_space(env_params).sample(act_rng), env_params)
        obses.append(obs)

    lambda_idx_to_test = 2
    obses = np.stack(obses)

    obs_for_lambda = obses[..., lambda_idx_to_test::n_lambdas]
    obs_for_lambda_without_action = obs_for_lambda[..., :-1]

    assert np.allclose(obs_for_lambda_without_action.sum(axis=-1), 1.)

    assert done.item()

    print()


def test_image_simple_chain():
    n = 10
    rng = jax.random.PRNGKey(2025)

    unwrapped_env = FullyObservableImageSimpleChain(n=n)
    env_params = unwrapped_env.default_params

    env = ActionConcatWrapper(unwrapped_env)
    env = TraceFeaturesWrapper(env, trace_in_obs=True)

    n_lambdas = len(env.lambdas)

    rng, reset_rng = jax.random.split(rng)
    obs, state = env.reset(reset_rng, env_params)
    obses = [obs]
    for i in range(n):
        rng, step_rng, act_rng = jax.random.split(rng, 3)
        obs, state, r, done, _ = env.step(step_rng, state, env.action_space(env_params).sample(act_rng), env_params)
        obses.append(obs)

    lambda_idx_to_test = 2
    obses = np.stack(obses)

    obs_for_lambda = obses[..., lambda_idx_to_test::n_lambdas]
    obs_for_lambda_without_action = obs_for_lambda[..., :-1]

    assert np.allclose(obs_for_lambda_without_action.sum(axis=-1), 1.)

    assert done.item()

    print()





if __name__ == '__main__':
    # test_simple_chain()
    test_image_simple_chain()
#     import numpy as np
#     lambdas = np.array([0., 0.3, 0.5, 0.9])
#     obs = np.ones((4, 4))
#     done = np.array(0)
#     # reset
#     trace_features = obs[..., None].repeat(len(lambdas), axis=-1)
#
#     # step
#     obs = np.array([[0, 1, 0, 1],
#                     [1, 0, 1, 0],
#                     [0, 1, 0, 1],
#                     [1, 0, 1, 0]])
#     leading_dims = (1,) * len(obs.shape)
#     lambdas = np.broadcast_to(lambdas, leading_dims + lambdas.shape)
#
#     next_trace = (1 - done) * lambdas * trace_features + (1 - lambdas) * obs[..., None]
#
#
#     print()
