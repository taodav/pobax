from functools import partial
from collections import deque
from dataclasses import replace
import inspect
from time import time

import chex
import flax.training.train_state
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint

from pobax.algos.ppo import PPO, env_step, calculate_gae, filter_period_first_dim, Transition
from pobax.config import QRPPOHyperparams
from pobax.envs import get_env
from pobax.envs.wrappers.gymnax import LogEnvState
from pobax.models import ScannedRNN
from pobax.models.actor_critic import QuantileActorCritic
from pobax.utils.file_system import get_results_path, numpyify


class QRPPO(PPO):
    def __init__(self, network,
                 double_critic: bool = False,
                 ld_weight: float = 0.,
                 alpha: float = 0.,
                 vf_coeff: float = 0.,
                 entropy_coeff: float = 0.01,
                 clip_eps: float = 0.2,
                 n_atoms: int = 51,
                 kappa: float = 1.,
                 quantile_entropy_coeff: float = 0.):
        super().__init__(network, double_critic, ld_weight, alpha, vf_coeff, entropy_coeff, clip_eps)
        self.n_atoms = n_atoms
        self.atoms = jnp.arange(self.n_atoms)
        self.kappa = kappa
        self.quantile_entropy_coeff = quantile_entropy_coeff

    def loss(self, params, init_hstate, traj_batch, gae, targets):
        # RERUN NETWORK
        _, pi, quantiles = self.network.apply(
            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
        )
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE QUANTILE VALUE LOSS
        quantiles_clipped = traj_batch.value + (
                quantiles - traj_batch.value
        ).clip(-self.clip_eps, self.clip_eps)

        td_err = targets[..., None, :] - quantiles[..., None]
        clipped_td_err = targets[..., None, :] - quantiles_clipped[..., None]  # b x n_atoms x n_atoms

        def loss_over_err(err: jnp.ndarray):
            huber_loss = ((jnp.abs(err) <= self.kappa).astype(float) * 0.5 * (err ** 2)
                          + (jnp.abs(err) > self.kappa).astype(float) * self.kappa
                          * (jnp.abs(err) - 0.5 * self.kappa)
                          )

            # get quantile mid points
            tau_hat = (self.atoms + 0.5) / self.atoms.shape[0]
            all_idxes = list(range(len(err.shape)))
            dims_to_expand = [i for i in all_idxes if i != all_idxes[-2]]
            tau_hat_expanded = jnp.expand_dims(tau_hat, dims_to_expand)
            tau_bellman_diff = jnp.abs(
                tau_hat_expanded - (err < 0).astype(float)
            )
            quantile_huber_loss = tau_bellman_diff * huber_loss
            loss = jnp.sum(jnp.mean(quantile_huber_loss, axis=-1), axis=-1)
            return loss

        value_loss = loss_over_err(td_err)
        clipped_value_loss = loss_over_err(clipped_td_err)
        value_loss = (
            jnp.maximum(value_loss, clipped_value_loss).mean()
        )

        # Quantile lambda discrepancy loss.
        # the mean l2 distance between quantiles is the Wasserstein-2 distance, as per
        # https://stats.stackexchange.com/questions/465229/measuring-the-distance-between-two-probability-measures-using-quantile-functions
        wasserstein_2_dist = 0
        if self.double_critic:
            wasserstein_2_dist = (jnp.square(quantiles[..., 0, :] - quantiles[..., 1, :])).mean()

        # Minimizing quantile entropy
        quantile_entropy_loss = jnp.log(quantiles[..., 1:] - quantiles[..., :-1])

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)

        # which advantage do we use to update our policy?
        if self.double_critic:
            gae = (self.alpha * gae[..., 0] +
                   (1 - self.alpha) * gae[..., 1])
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                    )
                * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
                loss_actor
                + self.vf_coeff * value_loss
                + self.ld_weight * wasserstein_2_dist
                + self.quantile_entropy_coeff * quantile_entropy_loss
                - self.entropy_coeff * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)


def make_train(args: QRPPOHyperparams, rand_key: jax.random.PRNGKey):
    num_updates = (
            args.total_steps // args.num_steps // args.num_envs
    )
    args.minibatch_size = (
            args.num_envs * args.num_steps // args.num_minibatches
    )
    env_key, rand_key = jax.random.split(rand_key)
    env, env_params = get_env(args.env, env_key,
                              gamma=args.gamma,
                              normalize_image=False,
                              perfect_memory=args.perfect_memory,
                              action_concat=args.action_concat)

    if hasattr(env, 'gamma'):
        args.gamma = env.gamma

    assert hasattr(env_params, 'max_steps_in_episode')

    double_critic = args.double_critic

    n_atoms = args.n_atoms if isinstance(args, QRPPOHyperparams) else None
    network = QuantileActorCritic(env.action_space(env_params),
                          memoryless=args.memoryless,
                          double_critic=args.double_critic,
                          hidden_size=args.hidden_size,
                          n_atoms=n_atoms)

    steps_filter = partial(filter_period_first_dim, n=args.steps_log_freq)
    update_filter = partial(filter_period_first_dim, n=args.update_log_freq)

    # Used for vmapping over our double critic.
    transition_axes_map = Transition(
        None, None, 2, None, None, None, None
    )

    _calculate_gae = calculate_gae
    if double_critic:
        # last_val is index 1 here b/c we squeezed earlier.
        _calculate_gae = jax.vmap(calculate_gae,
                                  in_axes=[transition_axes_map, 1, None, 0, None],
                                  out_axes=2)

    def train(vf_coeff, ld_weight, alpha, lambda1, lambda0, lr, rng):
        agent = QRPPO(network, double_critic=double_critic, ld_weight=ld_weight, alpha=alpha, vf_coeff=vf_coeff,
                      clip_eps=args.clip_eps, entropy_coeff=args.entropy_coeff,
                      quantile_entropy_coeff=args.quantile_entropy_coeff)

        # initialize functions
        _env_step = partial(env_step, agent=agent, env=env, env_params=env_params)

        gae_lambda = jnp.array(lambda0)
        if double_critic:
            gae_lambda = jnp.array([lambda0, lambda1])

        def linear_schedule(count):
            frac = (
                    1.0
                    - (count // (args.num_minibatches * args.update_epochs))
                    / num_updates
            )
            return lr * frac


        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, args.num_envs, *env.observation_space(env_params).shape)
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
                optax.adam(lr, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=agent.network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)

        if not args.env.startswith("craftax"):
            # We first need to populate our LogEnvState stats.
            rng, _rng = jax.random.split(rng)
            init_rng = jax.random.split(_rng, args.num_envs)
            init_obsv, init_env_state = env.reset(init_rng, env_params)
            init_init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)

            init_runner_state = (
                train_state,
                env_state,
                init_obsv,
                jnp.zeros(args.num_envs, dtype=bool),
                init_init_hstate,
                _rng,
            )

            starting_runner_state, _ = jax.lax.scan(
                _env_step, init_runner_state, None, env_params.max_steps_in_episode
            )

            def recursive_replace(env_state, new_env_state, names):
                if not isinstance(env_state, LogEnvState):
                    return replace(env_state, env_state=recursive_replace(env_state.env_state, new_env_state.env_state, names))
                new_log_vals = {name: getattr(new_env_state, name) for name in names}
                return replace(env_state, **new_log_vals)

            replace_field_names = ['returned_episode_returns', 'returned_discounted_episode_returns', 'returned_episode_lengths']
            env_state = recursive_replace(env_state, starting_runner_state[1], replace_field_names)
            # jax.debug.print("Training starting: {}", time())

        # TRAIN LOOP
        def _update_step(runner_state, i):
            # COLLECT TRAJECTORIES
            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(args.num_steps), args.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
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

            rng = update_state[-1]
            if args.debug:

                def callback(info):
                    timesteps = (
                            info["timestep"][info["returned_episode"]] * args.num_envs
                    )
                    if args.show_discounted:
                        show_str = "avg discounted return"
                        avg_return_values = jnp.mean(info["returned_discounted_episode_returns"][info["returned_episode"]])
                    else:
                        show_str = "avg episodic return"
                        avg_return_values = jnp.mean(info["returned_episode_returns"][info["returned_episode"]])

                    if len(timesteps) > 0:
                        print(
                            f"timesteps={timesteps[0]} - {timesteps[-1]}, {show_str}={avg_return_values:.2f}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((args.num_envs), dtype=bool),
            init_hstate,
            _rng,
        )

        # returned metric has an extra dimension.
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(num_updates), num_updates
        )

        # save metrics only every update_log_freq
        metric = jax.tree.map(update_filter, metric)

        # TODO: offline eval here.
        final_train_state = runner_state[0]

        if not args.env.startswith("craftax"):
            reset_rng = jax.random.split(_rng, args.num_envs)
        else:
            reset_rng, _rng = jax.random.split(_rng)
        eval_obsv, eval_env_state = env.reset(reset_rng, env_params)

        eval_init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)

        eval_runner_state = (
            final_train_state,
            eval_env_state,
            eval_obsv,
            jnp.zeros((args.num_envs), dtype=bool),
            eval_init_hstate,
            _rng,
        )

        # COLLECT EVAL TRAJECTORIES
        eval_runner_state, eval_traj_batch = jax.lax.scan(
            _env_step, eval_runner_state, None, env_params.max_steps_in_episode
        )
        # res = {"runner_state": runner_state, "metric": metric}
        res = {"runner_state": runner_state, "metric": metric, 'final_eval_metric': eval_traj_batch.info}

        return res

    return train


if __name__ == "__main__":
    # jax.disable_jit(True)
    args = QRPPOHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    rng = jax.random.PRNGKey(args.seed)
    make_train_rng, rng = jax.random.split(rng)
    rngs = jax.random.split(rng, args.n_seeds)
    train_fn = make_train(args, make_train_rng)
    train_args = list(inspect.signature(train_fn).parameters.keys())

    vmaps_train = train_fn
    swept_args = deque()

    # we need to go backwards, since JAX returns indices
    # in the order in which they're vmapped.
    for i, arg in reversed(list(enumerate(train_args))):
        dims = [None] * len(train_args)
        dims[i] = 0
        vmaps_train = jax.vmap(vmaps_train, in_axes=dims)
        if arg == 'rng':
            swept_args.appendleft(rngs)
        else:
            assert hasattr(args, arg)
            swept_args.appendleft(getattr(args, arg))

    train_jit = jax.jit(vmaps_train)
    t = time()
    out = train_jit(*swept_args)
    new_t = time()
    total_runtime = new_t - t
    print('Total runtime:', total_runtime)

    # our final_eval_metric returns max_num_steps.
    # we can filter that down by the max episode length amongst the runs.
    final_eval = out['final_eval_metric']
    final_train_state = out['runner_state'][0]

    # # the +1 at the end is to include the done step
    # largest_episode = final_eval['returned_episode'].argmax(axis=-2).max() + 1

    # def get_first_n_filter(x):
    #     return x[..., :largest_episode, :]
    # out['final_eval_metric'] = jax.tree.map(get_first_n_filter, final_eval)

    final_train_state = out['runner_state'][0]
    if not args.save_runner_state:
        del out['runner_state']

    results_path = get_results_path(args, return_npy=False)  # returns a results directory

    all_results = {
        'argument_order': train_args,
        'out': out,
        'args': args.as_dict(),
        'total_runtime': total_runtime,
        'final_train_state': final_train_state
    }

    all_results = jax.tree.map(numpyify, all_results)

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}")
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)
    print("Done.")
