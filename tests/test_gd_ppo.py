"""
To test LD:
check T-Maze optimal policy converges and that values are the same.

To test GVFs:
Have observation signals down a single corridor at different time steps.
The initial GVF should correspond to the discounted time-to-reach each signal

To test hidden-state dependent gammas:
TODO
"""
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np

from pobax.algos.gd_ppo import GDPPO, GDPPOHyperparams, make_train, env_step
from pobax.envs import get_env
from pobax.models import ScannedRNN
from pobax.models.actor_critic import CumulantNetwork, HangmanNetwork, ActorCritic


def load_vars(args: GDPPOHyperparams, rng: jax.random.PRNGKey):
    rng, env_rng = jax.random.split(rng)
    env, env_params = get_env(args.env, env_rng,
                              gamma=args.gamma,
                              normalize_image=False,
                              perfect_memory=args.perfect_memory,
                              action_concat=args.action_concat,
                              reward_concat=args.reward_concat)

    cumulant_size = None
    if args.cumulant_type == 'random_proj_hs' or args.cumulant_type == 'random_proj_obs':
        # randomly projected hidden state
        cumulant_size = args.cumulant_map_size
    elif args.cumulant_type == 'rew':
        cumulant_size = 1
    elif args.cumulant_type == 'hs' or args.cumulant_type == 'hs_diff':
        # Raw hidden state and raw hs difference
        cumulant_size = args.hidden_size
    elif args.cumulant_type == 'hs_rew':
        cumulant_size = args.cumulant_map_size + 1
    elif args.cumulant_type == 'obs' or args.cumulant_type == 'obs_diff':
        obs_shape = env.observation_space(env_params).shape
        assert len(obs_shape) <= 1, 'no support for images for obs cumulant'
        cumulant_size = obs_shape[0]
    elif args.cumulant_type == 'enc_obs':
        cumulant_size = args.hidden_size

    network = ActorCritic(env.action_space(env_params),
                             memoryless=args.memoryless,
                             double_critic=args.double_critic,
                             hidden_size=args.hidden_size,
                             cumulant_size=cumulant_size)

    cumulant_network = CumulantNetwork(cumulant_size=args.cumulant_map_size,)
    hangman_network = HangmanNetwork(gamma=args.gamma,
                                     gamma_type=args.gamma_type,
                                     gamma_max=args.gamma_max,
                                     gamma_min=args.gamma_min)

    agent = GDPPO(network, cumulant_network, hangman_network,
                  double_critic=args.double_critic,
                  ld_weight=args.ld_weight[0],
                  alpha=args.alpha[0],
                  vf_coeff=args.vf_coeff[0],
                  clip_eps=args.clip_eps,
                  entropy_coeff=args.entropy_coeff,
                  ld_exploration_bonus_scale=args.ld_exploration_bonus_scale)

    return env, env_params, agent

def run_n_steps_with_args(args: GDPPOHyperparams, n: int = 10):
    rng = jax.random.PRNGKey(args.seed)
    make_rng, train_rng, env_rng, rng = jax.random.split(rng, 4)
    train_fn = jax.jit(make_train(args, rng))
    # train_fn = make_train(args, rng)

    res = train_fn(args.vf_coeff[0], args.ld_weight[0], args.alpha[0], args.lambda1[0], args.lambda0[0], args.lr[0], train_rng)

    env, env_params, agent = load_vars(args, env_rng)
    ts = res['runner_state'][0]

    # initialize functions
    _env_step = partial(env_step, agent=agent, env=env, env_params=env_params, cumulant_type=args.cumulant_type)

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


def test_rew_cumulant():
    args = GDPPOHyperparams().from_dict({
        'env': 'fully_observable_simplechain',
        'gamma': 0.9,
        'hidden_size': 16,
        'cumulant_type': 'rew',
        'total_steps': int(1e6),
        'seed': 2025,
        'debug': True
    })

    res, _, traj_batch = run_n_steps_with_args(args, n=10)
    traj_batch = jax.tree.map(lambda x: np.array(x), traj_batch)
    ground_truth_values = ((0.9 ** np.arange(10))[::-1])[..., None]
    assert np.allclose(traj_batch.cumulant_value[:, 0], ground_truth_values)


def test_ld():
    args = GDPPOHyperparams().from_dict({
        'env': 'fully_observable_simplechain',
        'gamma': 0.9,
        'hidden_size': 16,
        'double_critic': True,
        'cumulant_type': 'rew',
        'ld_weight': [0.25],
        'total_steps': int(1e6),
        'seed': 2025,
        'debug': True
    })
    res, _, traj_batch = run_n_steps_with_args(args, n=10)
    traj_batch = jax.tree.map(lambda x: np.array(x), traj_batch)

    ground_truth_values = ((0.9 ** np.arange(10))[::-1])[..., None].repeat(2, axis=-1)
    assert np.allclose(traj_batch.cumulant_value[:, 0, :, 0], ground_truth_values)


def test_obs_sr():
    args = GDPPOHyperparams().from_dict({
        'env': 'fully_observable_simplechain',
        'gamma': 0.9,
        'hidden_size': 16,
        # 'double_critic': True,
        'cumulant_type': 'obs',
        # 'ld_weight': [0.25],
        'total_steps': int(1e6),
        'seed': 2025,
        'debug': True
    })
    res, _, traj_batch = run_n_steps_with_args(args, n=10)
    traj_batch = jax.tree.map(lambda x: np.array(x), traj_batch)
    ground_truth_values = (0.9 ** np.arange(10))[::-1]

    obs_cumulants = []
    for i in range(10):
        cumulant = np.zeros(10)
        cumulant[:i + 1] = ground_truth_values[-(i + 1):]
        obs_cumulants.append(cumulant)
    obs_cumulants = np.stack(obs_cumulants, axis=0)[::-1, ::-1]
    action_cumulant = obs_cumulants.sum(axis=-1, keepdims=True)

    # we subtract one here b/c action at reset is 0.
    action_cumulant[0] -= 1

    cumulants = np.concatenate((obs_cumulants, action_cumulant), axis=-1)

    # leave our reward vector here, since it's always 0.
    assert np.allclose(cumulants, traj_batch.cumulant_value[:, 0, :-1], atol=1e-3)


def test_hangman_gamma():
    args = GDPPOHyperparams().from_dict({
        'env': 'fully_observable_simplechain',
        'gamma': 0.9,
        'hidden_size': 16,
        'cumulant_type': 'rew',
        'gamma_type': 'nn_gamma_sigmoid',
        'total_steps': int(1e6),
        'seed': 2025,
        'debug': True
    })
    res, _, traj_batch = run_n_steps_with_args(args, n=10)
    traj_batch = jax.tree.map(lambda x: np.array(x), traj_batch)

    gammas = traj_batch.hangman
    up_tri = np.triu(np.ones((10, 10)))
    hangman_gt_returns = ((up_tri * gammas[:, 0]) + (up_tri == 0)).prod(axis=-1)

    predicted_values = traj_batch.cumulant_value[:, 0]

    assert np.allclose(predicted_values[:, 0], hangman_gt_returns)



if __name__ == "__main__":
    # jax.disable_jit(True)
    # test_rew_cumulant()
    # test_ld()
    test_obs_sr()
    # test_hangman_gamma()
