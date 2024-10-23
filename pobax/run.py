"""
Runs without a jax environment
"""
from functools import partial
from time import time

import chex
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax

from pobax.config import PPOHyperparams
from pobax.envs import get_pixel_env, get_env
from pobax.models import get_network_fn, ScannedRNN, get_gymnax_network_fn
from pobax.algos.ppo import PPO, calculate_gae, Transition


def mean_map_metrics(i, key: str):
    metrics = jax.tree.map(lambda *leaves: jnp.stack(leaves), *i)
    return metrics[key].mean()


def make_update(args: PPOHyperparams, rand_key: chex.PRNGKey):

    # env, env_params = get_pixel_env(args.env, gamma=args.gamma)
    #

    # network_fn, obs_shape, action_size = get_network_fn(env, memoryless=args.memoryless)

    # DEBUG
    env_key, rand_key = jax.random.split(rand_key)
    env, env_params = get_env(args.env, env_key,
                              gamma=args.gamma,
                              action_concat=args.action_concat)

    network_fn, action_size = get_gymnax_network_fn(env, env_params)

    network = network_fn(action_size,
                         double_critic=args.double_critic,
                         hidden_size=args.hidden_size)
    agent = PPO(network, args.double_critic,
                ld_weight=args.ld_weight, alpha=args.alpha, vf_coeff=args.vf_coeff,
                clip_eps=args.clip_eps, entropy_coeff=args.entropy_coeff)

    # Used for vmapping over our double critic.
    transition_axes_map = Transition(
        None, None, 2, None, None, None, None
    )

    gae_lambda = jnp.array(args.lambda0)
    _calculate_gae = jax.jit(calculate_gae)
    if args.double_critic:
        # last_val is index 1 here b/c we squeezed earlier.
        _calculate_gae = jax.vmap(calculate_gae,
                                 in_axes=[transition_axes_map, 1, None, 0],
                                 out_axes=2)
        gae_lambda = jnp.array([args.lambda0, args.lambda1])

    @jax.jit
    def _update_step(rng, transitions, last_obs, last_done, initial_hstate, last_hstate, train_state):
        # here we do everything post collecting
        traj_batch = jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *transitions)

        # CALCULATE ADVANTAGE
        ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
        _, _, last_val = network.apply(train_state.params, last_hstate, ac_in)
        last_val = last_val.squeeze(0)

        advantages, targets = calculate_gae(traj_batch, last_val, last_done, gae_lambda, args.gamma)

        # UPDATE NETWORK
        def _update_minbatch(train_state, batch_info):
            init_hstate, traj_batch, advantages, targets = batch_info

            grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
            total_loss, grads = grad_fn(
                train_state.params, init_hstate, traj_batch, advantages, targets
            )
            train_state = train_state.apply_gradients(grads=grads)
            return train_state, total_loss

        # SHUFFLE COLLECTED BATCH
        rng, _rng = jax.random.split(rng)
        permutation = jax.random.permutation(_rng, args.num_envs)
        init_hstate = initial_hstate[None, ...]
        batch = (init_hstate, traj_batch, advantages, targets)

        shuffled_batch = jax.tree.map(
            lambda x: jnp.take(x, permutation, axis=1), batch
        )

        minibatches = jax.tree.map(
            lambda x: jnp.swapaxes(
                jnp.reshape(
                    x,
                    [x.shape[0], args.num_minibatches, -1]
                    + list(x.shape[2:]),
                    ),
                1,
                0,
            ),
            shuffled_batch,
        )

        train_state, total_loss = jax.lax.scan(
            _update_minbatch, train_state, minibatches
        )
        return train_state, total_loss
    return _update_step, agent, env, env_params


if __name__ == "__main__":
    # jax.disable_jit(True)
    args = PPOHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    args.num_updates = (
            args.total_steps // args.num_steps // args.num_envs
    )
    args.minibatch_size = (
            args.num_envs * args.num_steps // args.num_minibatches
    )

    # Check that we're not trying to vmap over hyperparams
    for k, v in args.as_dict().items():
        if isinstance(v, list) or isinstance(v, np.ndarray) or isinstance(v, jnp.ndarray):
            assert len(v) == 1, "Can't run with multiple hyperparams."
            setattr(args, k, v[0])

    map_metrics = jax.jit(partial(mean_map_metrics, key='returned_episode_returns'))

    def linear_schedule(count):
        frac = (
                1.0
                - (count // (args.num_minibatches * args.update_epochs))
                / args.num_updates
        )
        return args.lr * frac

    np.random.seed(args.seed)
    rng = jax.random.PRNGKey(args.seed)
    rng, _rng = jax.random.split(rng)

    update_step, agent, env, env_params = make_update(args, _rng)

    obs_shape = env.observation_space(env_params).shape

    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = (
        jnp.zeros(
            (1, args.num_envs, *obs_shape)
        ),
        jnp.zeros((1, args.num_envs)),
    )
    init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
    network_params = agent.network.init(_rng, init_hstate, init_x)
    if args.anneal_lr:
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(learning_rate=linear_schedule, eps=1e-5),
        )
    else:
        tx = optax.chain(
            optax.clip_by_global_norm(args.max_grad_norm),
            optax.adam(args.lr, eps=1e-5),
        )
    train_state = TrainState.create(
        apply_fn=agent.network.apply,
        params=network_params,
        tx=tx,
    )

    # COLLECT MINIBATCH
    rng, _rng = jax.random.split(rng)
    reset_keys = jax.random.split(_rng, args.num_envs)
    last_obs, env_state = env.reset(reset_keys, env_params)
    last_done = jnp.zeros(args.num_envs, dtype=bool)

    hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
    t = time()

    for update_num in range(args.num_updates):
        transitions = []
        infos = []
        init_hstate = hstate

        for i in range(args.num_steps):
            rng, _rng = jax.random.split(rng)
            value, action, log_prob, hstate = agent.act(_rng, train_state, hstate, last_obs, last_done)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, hstate.shape[0])
            obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
            transition = Transition(
                last_done, action, value, reward, log_prob, last_obs, info
            )
            infos.append(info)
            transitions.append(transition)
            last_obs, last_done = obsv, done

        # UPDATE FROM MINIBATCH
        update_rng, rng = jax.random.split(rng)
        train_state, step_loss = update_step(update_rng, transitions, last_obs, last_done, init_hstate, hstate, train_state)

        if args.debug:
            metric = map_metrics(infos)
            new_t = time()
            time_per_step = (new_t - t)
            print(
                f"Mean return for step {update_num * args.num_steps * args.num_envs}/{args.total_steps}, "
                f"avg episodic return={metric:.2f}, total_loss={step_loss[0].mean():.2f}, "
                f"Time per update: {time_per_step:.2f}"
            )
            t = new_t

    print()
