from functools import partial

from flax.training.train_state import TrainState
from gymnax.environments import spaces
import jax.numpy as jnp
import jax
import optax
import numpy as np

from pobax.algos.sf_ppo import SFPPO, env_step, calculate_gae, make_train
from pobax.config import SFPPOHyperparams
from pobax.envs import get_env
from pobax.models.network import ScannedRNN
from pobax.models.actor_critic import SFActorCritic


def load_vars(args: SFPPOHyperparams, rng: jax.random.PRNGKey):
    rng, env_rng = jax.random.split(rng)
    env, env_params = get_env(args.env, env_rng,
                              gamma=args.gamma,
                              normalize_image=False,
                              perfect_memory=args.perfect_memory,
                              action_concat=args.action_concat,
                              trace_lambdas=args.trace_lambdas if args.use_trace_features else None)

    network = SFActorCritic(env.action_space(env_params),
                            memoryless=args.memoryless,
                            double_critic=args.double_critic,
                            hidden_size=args.hidden_size)

    agent = SFPPO(network,
                  double_critic=args.double_critic,
                  ld_weight=args.ld_weight[0],
                  alpha=args.alpha[0],
                  vf_coeff=args.vf_coeff[0],
                  clip_eps=args.clip_eps, entropy_coeff=args.entropy_coeff,
                  discrep_over=args.discrep_over)

    return env, env_params, agent

def run_n_steps_with_args(args: SFPPOHyperparams, n: int = 10):
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
    return res, runner_state, agent, traj_batch

def test_predictions():
    args = SFPPOHyperparams().from_dict({
        'env': 'fully_observable_simplechain',
        'gamma': 0.9,
        'hidden_size': 16,
        'total_steps': int(1e6),
        'seed': 2025,
        'debug': True
    })
    res, runner_state, agent, traj_batch = run_n_steps_with_args(args, n=10)
    train_state = runner_state[0]
    traj_batch = jax.tree.map(lambda x: np.array(x), traj_batch)

    # Value predictions
    ground_truth_values = ((0.9 ** np.arange(10))[::-1])
    assert np.allclose(traj_batch.value[:, 0], ground_truth_values, atol=1e-6)

    # Reward predictions
    next_encoded_obs = agent.network.apply(train_state.params, traj_batch.next_obs,
                                           method=agent.network.get_encoding)
    rewards = agent.network.apply(train_state.params, next_encoded_obs,
                                  method=agent.network.get_reward)
    rewards = np.array(rewards)
    assert np.allclose(rewards[:, 0], traj_batch.reward[:, 0], atol=1e-6)

def test_discrep_predictions():
    args = SFPPOHyperparams().from_dict({
        'env': 'fully_observable_simplechain',
        'gamma': 0.9,
        'hidden_size': 16,
        'total_steps': int(1e6),
        'seed': 2025,
        'double_critic': True,
        'ld_weight': [0.5],
        # 'discrep_type': 'rew',
        'discrep_over': 'sf',
        'debug': True
    })
    res, runner_state, agent, traj_batch = run_n_steps_with_args(args, n=10)
    traj_batch = jax.tree.map(lambda x: np.array(x), traj_batch)

    ground_truth_values = ((0.9 ** np.arange(10))[::-1])[..., None]
    assert np.allclose(traj_batch.value[:, 0] - ground_truth_values, 0, atol=1e-5)


def test_env_grads():
    args = SFPPOHyperparams().from_dict({'env': 'tmaze_5',
                                       'memoryless': False,
                                       'hidden_size': 32})
    rand_key = jax.random.PRNGKey(2025)

    env_key, rand_key = jax.random.split(rand_key)
    env, env_params = get_env(args.env, env_key,
                              gamma=args.gamma,
                              normalize_image=False,
                              perfect_memory=args.perfect_memory,
                              action_concat=args.action_concat,
                              trace_lambdas=args.trace_lambdas if args.use_trace_features else None)

    network = SFActorCritic(env.action_space(env_params),
                            memoryless=args.memoryless,
                            double_critic=args.double_critic,
                            hidden_size=args.hidden_size)

    agent = SFPPO(network,
                  double_critic=args.double_critic,
                  clip_eps=args.clip_eps,
                  entropy_coeff=args.entropy_coeff)

    # initialize functions
    _env_step = partial(env_step, agent=agent, env=env, env_params=env_params)


    # INIT NETWORK
    rng, _rng = jax.random.split(rand_key)
    init_x = (
        jnp.zeros(
            (1, args.num_envs, *env.observation_space(env_params).shape)
        ),
        jnp.zeros((1, args.num_envs)),
    )
    init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
    network_params = agent.network.init(_rng, init_hstate, init_x)

    # INIT ENV
    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, args.num_envs)
    obsv, env_state = env.reset(reset_rng, env_params)

    tx = optax.chain(
        optax.clip_by_global_norm(args.max_grad_norm),
        optax.adam(args.lr, eps=1e-5),
    )
    train_state = TrainState.create(
        apply_fn=agent.network.apply,
        params=network_params,
        tx=tx,
    )
    init_runner_state = (
        train_state,
        env_state,
        obsv,
        jnp.zeros(args.num_envs, dtype=bool),
        init_hstate,
        _rng,
    )

    runner_state, traj_batch = jax.lax.scan(
        _env_step, init_runner_state, None, 32
    )

    # CALCULATE ADVANTAGE
    train_state, env_state, last_obs, last_done, hstate, rng = runner_state
    ac_in = (last_obs[jnp.newaxis, :], last_done[jnp.newaxis, :])
    _, _, last_val = network.apply(train_state.params, hstate, ac_in)
    last_val = last_val.squeeze(0)

    advantages, targets = calculate_gae(traj_batch, last_val, last_done, 0.9, args.gamma)

    # SF loss, freeze reward params
    actor_sf_grad_fn = jax.value_and_grad(agent.actor_sf_loss, has_aux=True)
    actor_sf_out, actor_sf_grads = actor_sf_grad_fn(
        train_state.params, init_hstate, traj_batch, advantages, targets
    )

    # Check that all reward grads are 0
    reward_sf_grads = jax.tree.flatten(actor_sf_grads['params']['r'])[0]
    for g in reward_sf_grads:
        assert jnp.allclose(g, 0)

    next_encoded_obs = agent.network.apply(train_state.params, traj_batch.next_obs,
                                           method=agent.network.get_encoding)

    # Reward loss, freeze other params
    reward_grad_fn = jax.value_and_grad(agent.reward_loss)
    reward_loss, reward_grads = reward_grad_fn(
        train_state.params, next_encoded_obs, traj_batch
    )

    for k, v in reward_grads['params'].items():
        if k != 'r':
            for g in jax.tree.flatten(v)[0]:
                assert jnp.allclose(g, 0)



def test_simple_grads():
    """
    Test that we have zero gradients for reward params in sf_loss,
    as well as zero gradients for all non-reward parameters for reward_loss.
    """
    rng = jax.random.PRNGKey(2025)

    network = SFActorCritic(spaces.Discrete(3),
                            hidden_size=8,
                            memoryless=True)

    init_x = (
        jnp.ones(
            (1, 4, 12)
        ),
        jnp.zeros((1, 4)),
    )
    init_hstate = ScannedRNN.initialize_carry(4, 8)

    network_params = network.init(rng, init_hstate, init_x)

    def sf_loss(params, hstate, x):
        rew_params = network_params['params']['r']
        rew_params = (rew_params['kernel'], rew_params['bias'])
        _, _, v, encoding = network.apply(params, hstate, x, rew_params, method=SFActorCritic.get_sf)
        return (v ** 2).sum(), encoding

    sf_grad_fn = jax.value_and_grad(sf_loss, has_aux=True)
    vals, sf_grads = sf_grad_fn(network_params, init_hstate, init_x)

    # Check that all reward grads are 0
    reward_grads = jax.tree.flatten(sf_grads['params']['r'])[0]
    for g in reward_grads:
        assert jnp.allclose(g, 0)

    _, encoding = vals

    def reward_loss(params, encoding):
        r = network.apply(params, encoding, method=SFActorCritic.get_reward)
        return (r ** 2).sum()

    reward_grad_fn = jax.value_and_grad(reward_loss)
    vals, rew_grads = reward_grad_fn(network_params, encoding)

    for k, v in rew_grads['params'].items():
        if k != 'r':
            for g in jax.tree.flatten(v)[0]:
                assert jnp.allclose(g, 0)


if __name__ == "__main__":
    # jax.disable_jit(True)
    # test_simple_grads()
    # test_env_grads()
    # test_predictions()
    test_discrep_predictions()
