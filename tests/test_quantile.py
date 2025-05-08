
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from pobax.algos.qr_ppo import QRPPO, QRPPOHyperparams, make_train
from pobax.algos.ppo import env_step
from pobax.envs import get_env
from pobax.models import ScannedRNN
from pobax.models.actor_critic import ActorCritic


def load_vars(args: QRPPOHyperparams, rng: jax.random.PRNGKey):
    rng, env_rng = jax.random.split(rng)
    env, env_params = get_env(args.env, env_rng,
                              gamma=args.gamma,
                              normalize_image=False,
                              perfect_memory=args.perfect_memory,
                              action_concat=args.action_concat)

    network = ActorCritic(env.action_space(env_params),
                          memoryless=args.memoryless,
                          double_critic=args.double_critic,
                          hidden_size=args.hidden_size,
                          n_atoms=args.n_atoms)

    agent = QRPPO(network,
                  double_critic=args.double_critic,
                  ld_weight=args.ld_weight[0],
                  alpha=args.alpha[0],
                  vf_coeff=args.vf_coeff[0],
                  clip_eps=args.clip_eps,
                  entropy_coeff=args.entropy_coeff,
                  n_atoms=args.n_atoms)

    return env, env_params, agent


def run_n_steps_with_args(args: QRPPOHyperparams, n: int = 10):
    rng = jax.random.PRNGKey(args.seed)
    make_rng, train_rng, env_rng, rng = jax.random.split(rng, 4)
    train_fn = jax.jit(make_train(args, rng))
    # train_fn = make_train(args, rng)

    res = train_fn(args.vf_coeff[0], args.ld_weight[0], args.alpha[0], args.lambda1[0], args.lambda0[0], args.lr[0], train_rng)

    env, env_params, agent = load_vars(args, env_rng)
    ts = res['runner_state'][0]

    # initialize functions
    _env_step = partial(env_step, agent=agent, env=env, env_params=env_params)

    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, args.num_envs)
    init_obsv, env_state = env.reset(reset_rng, env_params)

    init_runner_state = (
        ts,
        env_state,
        init_obsv,
        jnp.zeros(args.num_envs, dtype=bool),
        init_hstate,
        _rng,
    )

    runner_state, traj_batch = jax.lax.scan(
        _env_step, init_runner_state, None, n
    )
    return res, runner_state, traj_batch


def test_fo_value():
    args = QRPPOHyperparams().from_dict({
        'env': 'fully_observable_simplechain',
        'memoryless': True,
        'gamma': 0.9,
        'hidden_size': 16,
        'total_steps': int(1e6),
        'seed': 2025,
        'debug': True
    })

    res, _, traj_batch = run_n_steps_with_args(args, n=10)
    traj_batch = jax.tree.map(lambda x: np.array(x), traj_batch)
    ground_truth_values = ((0.9 ** np.arange(10))[::-1])
    assert np.allclose(traj_batch.value[:, 0].mean(axis=-1), ground_truth_values)


def test_rnn_value():
    args = QRPPOHyperparams().from_dict({
        'env': 'simplechain',
        'gamma': 0.9,
        'hidden_size': 16,
        'total_steps': int(1e6),
        'seed': 2025,
        'debug': True
    })

    res, _, traj_batch = run_n_steps_with_args(args, n=10)
    traj_batch = jax.tree.map(lambda x: np.array(x), traj_batch)
    ground_truth_values = ((0.9 ** np.arange(10))[::-1])
    assert np.allclose(traj_batch.value[:, 0].mean(axis=-1), ground_truth_values, atol=1e-3)


if __name__ == "__main__":
    # test_fo_value()
    test_rnn_value()
