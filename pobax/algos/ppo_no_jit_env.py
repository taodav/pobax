from functools import partial
from time import time

import chex
from flax.training import orbax_utils
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint

from pobax.config import PPOHyperparams
from pobax.envs import get_gym_env
from pobax.models import get_network_fn, ScannedRNN
from pobax.utils.file_system import get_results_path

from pobax.algos.ppo import PPO, Transition, calculate_gae, filter_period_first_dim, env_step


def make_update(args: PPOHyperparams, rand_key: jax.random.PRNGKey):
    args.minibatch_size = (
            args.num_envs * args.num_steps // args.num_minibatches
    )

    double_critic = args.double_critic

    env = get_gym_env(args.env, gamma=args.gamma, seed=args.seed, num_envs=args.num_envs,
                      image_size=args.image_size)

    network_fn, obs_shape, action_size = get_network_fn(env, memoryless=args.memoryless)

    network = network_fn(action_size,
                         double_critic=double_critic,
                         hidden_size=args.hidden_size)

    steps_filter = partial(filter_period_first_dim, n=args.steps_log_freq)

    # Used for vmapping over our double critic.
    transition_axes_map = Transition(
        None, None, 2, None, None, None, None
    )

    _calculate_gae = calculate_gae
    if double_critic:
        # last_val is index 1 here b/c we squeezed earlier.
        _calculate_gae = jax.vmap(calculate_gae,
                                  in_axes=[transition_axes_map, 1, None, 0],
                                  out_axes=2)

    agent = PPO(network, double_critic=double_critic, ld_weight=args.ld_weight, alpha=args.alpha,
                vf_coeff=args.vf_coeff,
                clip_eps=args.clip_eps, entropy_coeff=args.entropy_coeff)

    gae_lambda = jnp.array(args.lambda0)
    if double_critic:
        gae_lambda = jnp.array([args.lambda0, args.lambda1])


    # def train(rng):

    # TRAIN LOOP
    # def _update_step(runner_state, i):
    def _update_step(rng: chex.PRNGKey, train_state: TrainState,
                     initial_hstate: chex.Array,
                     hstate: chex.Array,
                     traj_batch: Transition,
                     last_obs, last_done):


        # CALCULATE ADVANTAGE
        # train_state, env_state, last_obs, last_done, hstate, rng = runner_state
        ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
        _, _, last_val = network.apply(train_state.params, hstate, ac_in)
        last_val = last_val.squeeze(0)

        advantages, targets = _calculate_gae(traj_batch, last_val, last_done, gae_lambda, args.gamma)

        # UPDATE NETWORK
        def _update_epoch(update_state, unused):
            def _update_minbatch(train_state, batch_info):
                init_hstate, traj_batch, advantages, targets = batch_info

                grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
                total_loss, grads = grad_fn(
                    train_state.params, init_hstate, traj_batch, advantages, targets
                )
                train_state = train_state.apply_gradients(grads=grads)
                return train_state, total_loss

            (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            ) = update_state

            # SHUFFLE COLLECTED BATCH
            rng, _rng = jax.random.split(rng)
            permutation = jax.random.permutation(_rng, args.num_envs)
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
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages,
                targets,
                rng,
            )
            return update_state, total_loss

        init_hstate = initial_hstate[None, :]  # TBH
        update_state = (
            train_state,
            init_hstate,
            traj_batch,
            advantages,
            targets,
            rng,
        )
        update_state, loss_info = jax.lax.scan(
            _update_epoch, update_state, None, args.update_epochs
        )
        train_state = update_state[0]

        # save metrics only every steps_log_freq
        metric = traj_batch.info
        metric = jax.tree.map(steps_filter, metric)

        return train_state, {'info': metric, 'loss': loss_info}

    return _update_step, agent, env


@jax.jit
def stack_list(lst):
    return jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *lst)


if __name__ == "__main__":
    # jax.disable_jit(True)
    # okay some weirdness here. NUM_ENVS needs to match with NUM_MINIBATCHES
    args = PPOHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    # Check that we're not trying to vmap over hyperparams
    for k, v in args.as_dict().items():
        if isinstance(v, list) or isinstance(v, np.ndarray) or isinstance(v, jnp.ndarray):
            assert len(v) == 1, "Can't run with multiple hyperparams."
            setattr(args, k, v[0])

    num_updates = (
            args.total_steps // args.num_steps // args.num_envs
    )

    rng = jax.random.PRNGKey(args.seed)
    make_train_rng, rng = jax.random.split(rng)
    update_fn, agent, env = make_update(args, make_train_rng)
    update_jit = jax.jit(update_fn)


    # INIT NETWORK
    rng, init_rng = jax.random.split(rng)
    init_x = (
        jnp.zeros(
            (1, args.num_envs, *env.observation_space.shape)
        ),
        jnp.zeros((1, args.num_envs)),
    )
    init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
    network_params = agent.network.init(init_rng, init_hstate, init_x)

    def linear_schedule(count):
        frac = (
                1.0
                - (count // (args.num_minibatches * args.update_epochs))
                / num_updates
        )
        return args.lr * frac

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

    # INIT ENV
    obsv, info = env.reset()

    hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
    last_obs = obsv
    last_done = jnp.zeros(args.num_envs, dtype=bool)

    all_metrics = []
    t = time()

    def take_steps(rng, env, train_state, hstate, last_obs, last_done,
                   num_steps: int):
        """
        Runs the train_state over env for num_steps
        """
        transitions = []
        for step in range(num_steps):
            rng, _rng = jax.random.split(rng)
            value, action, log_prob, hstate = agent.act(_rng, train_state, hstate, last_obs, last_done)

            # STEP ENV
            rng, _rng = jax.random.split(rng)
            rng_step = jax.random.split(_rng, hstate.shape[0])
            obsv, reward, done, trunc, info = env.step(action)

            # TODO: add logging for non-gymnax envs
            transition = Transition(
                last_done, action, value, reward, log_prob, last_obs, info
            )
            transitions.append(transition)
            last_obs, last_done = obsv, done
            # TODO: this assumes auto-reset is in place.

        traj_batch = stack_list(transitions)
        return env, traj_batch, hstate, last_obs, last_done

    for update_num in range(num_updates):
        # COLLECT TRAJECTORIES
        initial_hstate = hstate

        rng, _rng = jax.random.split(rng)
        env, traj_batch, hstate, last_obs, last_done = take_steps(_rng, env, train_state, hstate, last_obs, last_done,
                                                                  num_steps=args.num_steps)

        # Update
        rng, _rng = jax.random.split(rng)
        train_state, update_metrics = update_jit(_rng, train_state, initial_hstate, hstate, traj_batch, last_obs, last_done)
        metrics = update_metrics['info']
        total_loss, _ = update_metrics['loss']  # loss infos are all 4x4 b/c we have 4 minibatches and 4 envs
        all_metrics.append(metrics)

        info = traj_batch.info

        if args.debug:
            metric = metrics['returned_episode_returns'].mean()
            new_t = time()
            time_per_step = (new_t - t)
            print(
                f"Mean return for step {update_num * args.num_steps * args.num_envs}/{args.total_steps}, "
                f"avg episodic return={metric:.2f}, "
                f"total_loss={total_loss.mean():.2f}, "
                f"Time per update: {time_per_step:.2f}"
            )
            t = new_t

    final_train_state = train_state
    out = {'metrics': stack_list(all_metrics)}

    print("Evaluating trained policy")
    # final evaluation
    eval_res = []
    max_steps_in_episode = env.max_steps_in_episode if hasattr(env, 'max_steps_in_episode') else args.default_max_steps_in_episode
    # For now, we use num_eval_envs as a surrogate for num eval episodes
    for i in range(args.num_eval_envs):
        # INIT ENV
        obsv, info = env.reset()

        hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
        last_obs = obsv
        last_done = jnp.zeros(args.num_envs, dtype=bool)

        rng, _rng = jax.random.split(rng)
        _, eval_batch, final_hstate, last_obs, last_done = take_steps(_rng, env, final_train_state, hstate, last_obs, last_done,
                                                                      num_steps=max_steps_in_episode)
        eval_res.append({'rewards': eval_batch.reward, 'done': eval_batch.done, 'info': eval_batch.info})

    out['eval'] = stack_list(eval_res)
    update_filter = partial(filter_period_first_dim, n=args.update_log_freq)

    out = jax.tree.map(update_filter, out)

    if args.save_runner_state:
        out['train_state'] = train_state

    results_path = get_results_path(args, return_npy=False)  # returns a results directory

    all_results = {
        'out': out,
        'args': args.as_dict()
    }

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}")
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)

    print("Done.")
