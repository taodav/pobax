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
from pobax.envs import get_pixel_env
from pobax.models import get_network_fn, ScannedRNN
from pobax.algos.ppo import PPO, calculate_gae, Transition


def make_train(args: PPOHyperparams, rand_key: chex.PRNGKey):

    steps_per_update = (args.num_envs * args.num_steps)
    num_updates = args.total_steps // steps_per_update

    env, env_params = get_pixel_env(args.env, gamma=args.gamma)

    network_fn, obs_shape, action_size = get_network_fn(env, memoryless=args.memoryless)

    network = network_fn(action_size,
                         double_critic=args.double_critic,
                         hidden_size=args.hidden_size)
    agent = PPO(network, args.double_critic,
                ld_weight=args.ld_weight, alpha=args.alpha, vf_coeff=args.vf_coeff,
                clip_eps=args.clip_eps, entropy_coeff=args.entropy_coeff)

    # Calculate num updates and minibatch size
    num_updates = (
            args.total_steps // args.num_steps // args.num_envs
    )
    args.minibatch_size = (
            args.num_envs * args.num_steps // args.num_minibatches
    )

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

    def linear_schedule(count):
        frac = (
                1.0
                - (count // (args.num_minibatches * args.update_epochs))
                / num_updates
        )
        return args.lr * frac


    # INIT NETWORK
    rng, _rng = jax.random.split(rand_key)
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

    # TODO: logging

    # COLLECT MINIBATCH
    # last_obs, info = env.reset()
    rng, _rng = jax.random.split(rng)
    reset_keys = jax.random.split(_rng, args.num_envs)
    last_obs, env_state = env.reset(reset_keys, env_params)
    last_done = jnp.zeros(args.num_envs, dtype=bool)

    hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
    t = time()
    new_t = t

    for update_num in range(num_updates):
        transitions = []
        infos = []
        init_hstate = hstate

        for i in range(args.num_steps):
            act_rng, step_rng, rng = jax.random.split(rng, 3)
            value, action, log_prob, hstate = agent.act(act_rng, train_state, hstate, last_obs, last_done)
            step_rngs = jax.random.split(step_rng, args.num_envs)
            obs, env_state, reward, dones, info = env.step(step_rngs, env_state, action, env_params)
            transitions.append(Transition(
                last_done, action, value, reward, log_prob, last_obs
            ))
            infos.append(info)
            last_obs, last_done = obs, dones

        # UPDATE FROM MINIBATCH
        @jax.jit
        def _update_step(rng, transitions, last_obs, last_done, starting_hstate, train_state):
            # here we do everything post collecting
            traj_batch = jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *transitions)

            # CALCULATE ADVANTAGE AND TARGETS
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, starting_hstate, ac_in)
            last_val = last_val.squeeze(0)

            advantages, targets = calculate_gae(traj_batch, last_val, last_done, gae_lambda, args.gamma)

            # UPDATE NETWORK
            def _update_minbatch(train_state, batch_info):
                starting_hstate, traj_batch, advantages, targets = batch_info

                starting_hstate = starting_hstate[None, :]  # TBH
                grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, starting_hstate, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss


            # SHUFFLE COLLECTED BATCH
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, args.num_envs)
            starting_hstate = starting_hstate[None, ...]
            batch = (starting_hstate, traj_batch, advantages, targets)

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

        update_rng, rng = jax.random.split(rng)
        train_state, step_loss = _update_step(rng, transitions, last_obs, last_done, init_hstate, train_state)

        if args.debug:
            def mean_map_metrics(i, key: str):
                metrics = jax.tree.map(lambda *leaves: jnp.stack(leaves), *i)
                return metrics[key].mean()

            map_metrics = jax.jit(partial(mean_map_metrics, key='returned_episode_returns'))
            metric = map_metrics(infos)
            new_t = time()
            time_per_step = (new_t - t)
            print(
                f"Mean return for step {update_num * steps_per_update}/{args.total_steps}, "
                f"avg episodic return={metric:.2f}, total_loss={step_loss[0].mean():.2f}, "
                f"Time per update: {time_per_step:.2f}"
            )
            t = new_t

        # finished_timestep_infos = [jax.tree.map(lambda *leaves: np.stack(leaves), *[ffinfo for ffinfo in finfo])
        #                         for finfo in
        #                             [inf['final_info'] for inf in infos if 'final_info' in inf]
        #                       ]
        # if finished_timestep_infos:
        #     returns = jnp.array([inf['returned_episode_return'] for inf in finished_timestep_infos])
        #     print(f"Mean returns for step {update_num * steps_per_update}/{args.total_steps}: {returns.mean()}, total loss: {step_loss[0].mean()}")


        # # save metrics only every steps_log_freq
        # metric = traj_batch.info
        # metric = jax.tree.map(steps_filter, metric)
        #
        # rng = update_state[-1]
        # if args.debug:
        #
        #     def callback(info):
        #         timesteps = (
        #                 info["timestep"][info["returned_episode"]] * args.num_envs
        #         )
        #         avg_return_values = jnp.mean(info["returned_episode_returns"][info["returned_episode"]])
        #         if len(timesteps) > 0:
        #             print(
        #                 f"timesteps={timesteps[0]} - {timesteps[-1]}, avg episodic return={avg_return_values:.2f}"
        #             )
        #
        #     jax.debug.callback(callback, metric)
        #
        # runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
        #
        # return runner_state, metric


if __name__ == "__main__":
    # jax.disable_jit(True)
    args = PPOHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    # Check that we're not trying to vmap over hyperparams
    for k, v in args.as_dict().items():
        if isinstance(v, list) or isinstance(v, np.ndarray) or isinstance(v, jnp.ndarray):
            assert len(v) == 1, "Can't run with multiple hyperparams."
            setattr(args, k, v[0])

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    make_train(args, key)

    print()
