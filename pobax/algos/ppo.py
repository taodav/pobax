from typing import NamedTuple

from collections import deque
from dataclasses import replace
from functools import partial
import inspect


from flax.training.train_state import TrainState
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint

from pobax.config import PPOHyperparams
from pobax.envs import get_env
from pobax.envs.wrappers import LogEnvState
from pobax.models import get_network_fn, ScannedRNN
from pobax.utils.file_system import get_results_path


class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray


def env_step(runner_state, unused, network, env, env_params):
    train_state, env_state, last_obs, last_done, hstate, rng = runner_state
    rng, _rng = jax.random.split(rng)

    # SELECT ACTION
    ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
    hstate, pi, value = network.apply(train_state.params, hstate, ac_in)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)
    value, action, log_prob = (
        value.squeeze(0),
        action.squeeze(0),
        log_prob.squeeze(0),
    )

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, hstate.shape[0])
    obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
    transition = Transition(
        last_done, action, value, reward, log_prob, last_obs, info
    )
    runner_state = (train_state, env_state, obsv, done, hstate, rng)
    return runner_state, transition



def filter_period_first_dim(x, n: int):
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        return x[::n]


def make_train(config: dict, rand_key: jax.random.PRNGKey):
    config["NUM_UPDATES"] = (
            config["TOTAL_TIMESTEPS"] // config["NUM_STEPS"] // config["NUM_ENVS"]
    )
    config["MINIBATCH_SIZE"] = (
            config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )
    env_key, rand_key = jax.random.split(rand_key)
    env, env_params = get_env(config['ENV_NAME'], env_key,
                                     gamma=config["GAMMA"],
                                     action_concat=config["ACTION_CONCAT"])

    if hasattr(env, 'gamma'):
        config['GAMMA'] = env.gamma

    assert hasattr(env_params, 'max_steps_in_episode')

    memoryless = config["MEMORYLESS"]

    network_fn, action_size = get_network_fn(env, env_params, memoryless=memoryless)

    network = network_fn(action_size,
                         hidden_size=config['HIDDEN_SIZE'])

    steps_filter = partial(filter_period_first_dim, n=config['STEPS_LOG_FREQ'])
    update_filter = partial(filter_period_first_dim, n=config['UPDATE_LOG_FREQ'])

    # Used for vmapping over our double critic.
    transition_axes_map = Transition(
        None, None, 2, None, None, None, None
    )

    _env_step = partial(env_step, network=network, env=env, env_params=env_params)

    def train(vf_coeff, lambda0, lr, rng):
        def linear_schedule(count):
            frac = (
                    1.0
                    - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
                    / config["NUM_UPDATES"]
            )
            return lr * frac


        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_x = (
            jnp.zeros(
                (1, config["NUM_ENVS"], *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, config["NUM_ENVS"])),
        )
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config['HIDDEN_SIZE'])
        network_params = network.init(_rng, init_hstate, init_x)
        if config["ANNEAL_LR"]:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(learning_rate=linear_schedule, eps=1e-5),
            )
        else:
            tx = optax.chain(
                optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
                optax.adam(lr, eps=1e-5),
            )
        train_state = TrainState.create(
            apply_fn=network.apply,
            params=network_params,
            tx=tx,
        )

        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, config["NUM_ENVS"])
        obsv, env_state = env.reset(reset_rng, env_params)
        init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config['HIDDEN_SIZE'])

        # We first need to populate our LogEnvState stats.
        rng, _rng = jax.random.split(rng)
        init_rng = jax.random.split(_rng, config["NUM_ENVS"])
        init_obsv, init_env_state = env.reset(init_rng, env_params)
        init_init_hstate = ScannedRNN.initialize_carry(config["NUM_ENVS"], config['HIDDEN_SIZE'])

        init_runner_state = (
            train_state,
            env_state,
            init_obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
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

        # TRAIN LOOP
        def _update_step(runner_state, i):
            # COLLECT TRAJECTORIES
            initial_hstate = runner_state[-2]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(config["NUM_STEPS"]), config["NUM_STEPS"]
            )

            # CALCULATE ADVANTAGE
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)
            def _calculate_gae(traj_batch, last_val, last_done, gae_lambda):
                def _get_advantages(carry, transition):
                    gae, next_value, next_done, gae_lambda = carry
                    done, value, reward = transition.done, transition.value, transition.reward
                    delta = reward + config["GAMMA"] * next_value * (1 - next_done) - value
                    gae = delta + config["GAMMA"] * gae_lambda * (1 - next_done) * gae
                    return (gae, value, done, gae_lambda), gae
                _, advantages = jax.lax.scan(_get_advantages,
                                             (jnp.zeros_like(last_val), last_val, last_done, gae_lambda),
                                             traj_batch, reverse=True, unroll=16)
                return advantages, advantages + traj_batch.value

            gae_lambda = jnp.array(lambda0)
            advantages, targets = _calculate_gae(traj_batch, last_val, last_done, gae_lambda)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets = batch_info

                    def _loss_fn(params, init_hstate, traj_batch, gae, targets):
                        # RERUN NETWORK
                        _, pi, value = network.apply(
                            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # CALCULATE VALUE LOSS
                        value_pred_clipped = traj_batch.value + (
                                value - traj_batch.value
                        ).clip(-config["CLIP_EPS"], config["CLIP_EPS"])
                        value_losses = jnp.square(value - targets)
                        value_losses_clipped = jnp.square(value_pred_clipped - targets)
                        value_loss = (
                            jnp.maximum(value_losses, value_losses_clipped).mean()
                        )

                        # CALCULATE ACTOR LOSS
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)

                        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
                        loss_actor1 = ratio * gae
                        loss_actor2 = (
                                jnp.clip(
                                    ratio,
                                    1.0 - config["CLIP_EPS"],
                                    1.0 + config["CLIP_EPS"],
                                    )
                                * gae
                        )
                        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
                        loss_actor = loss_actor.mean()
                        entropy = pi.entropy().mean()

                        total_loss = (
                                loss_actor
                                + vf_coeff * value_loss
                                - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (value_loss, loss_actor, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
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

                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch, advantages, targets)

                shuffled_batch = jax.tree.map(
                    lambda x: jnp.take(x, permutation, axis=1), batch
                )

                minibatches = jax.tree.map(
                    lambda x: jnp.swapaxes(
                        jnp.reshape(
                            x,
                            [x.shape[0], config["NUM_MINIBATCHES"], -1]
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
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]

            # save metrics only every steps_log_freq
            metric = traj_batch.info
            metric = jax.tree.map(steps_filter, metric)

            rng = update_state[-1]
            if config.get("DEBUG"):

                def callback(info):
                    timesteps = (
                            info["timestep"][info["returned_episode"]] * config["NUM_ENVS"]
                    )
                    avg_return_values = jnp.mean(info["returned_episode_returns"][info["returned_episode"]])
                    if len(timesteps) > 0:
                        print(
                            f"timesteps={timesteps[0]} - {timesteps[-1]}, avg episodic return={avg_return_values:.2f}"
                        )

                jax.debug.callback(callback, metric)

            runner_state = (train_state, env_state, last_obs, last_done, hstate, rng)
            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            env_state,
            obsv,
            jnp.zeros((config["NUM_ENVS"]), dtype=bool),
            init_hstate,
            _rng,
        )
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"]), config["NUM_UPDATES"]
        )

        # save metrics only every update_log_freq
        metric = jax.tree.map(update_filter, metric)

        # TODO: offline eval here.
        final_train_state = runner_state[0]

        reset_rng = jax.random.split(_rng, config["NUM_EVAL_ENVS"])
        eval_obsv, eval_env_state = env.reset(reset_rng, env_params)

        eval_init_hstate = ScannedRNN.initialize_carry(config["NUM_EVAL_ENVS"], config['HIDDEN_SIZE'])

        eval_runner_state = (
            final_train_state,
            eval_env_state,
            eval_obsv,
            jnp.zeros((config["NUM_EVAL_ENVS"]), dtype=bool),
            eval_init_hstate,
            _rng,
        )

        # COLLECT EVAL TRAJECTORIES
        eval_runner_state, eval_traj_batch = jax.lax.scan(
            _env_step, eval_runner_state, None, env_params.max_steps_in_episode
        )

        return {"runner_state": runner_state, "metric": metric, 'final_eval_metric': eval_traj_batch.info}

    return train


if __name__ == "__main__":
    # jax.disable_jit(True)
    # okay some weirdness here. NUM_ENVS needs to match with NUM_MINIBATCHES
    args = PPOHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    config = {
        "NUM_ENVS": 4,
        "NUM_EVAL_ENVS": 10,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": args.total_steps,
        "DEFAULT_MAX_STEPS_IN_EPISODE": args.default_max_steps_in_episode,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "MEMORYLESS": args.memoryless,
        "ACTION_CONCAT": args.action_concat,
        "CLIP_EPS": 0.2,
        "ENT_COEF": args.entropy_coeff,
        "VF_COEF": args.vf_coeff,
        "MAX_GRAD_NORM": 0.5,
        "HIDDEN_SIZE": args.hidden_size,
        "STEPS_LOG_FREQ": args.steps_log_freq,
        "UPDATE_LOG_FREQ": args.update_log_freq,
        "ENV_NAME": args.env,
        "ANNEAL_LR": True,
        "DEBUG": args.debug,
    }

    rng = jax.random.PRNGKey(args.seed)
    make_train_rng, rng = jax.random.split(rng)
    rngs = jax.random.split(rng, args.n_seeds)
    train_fn = make_train(config, make_train_rng)

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
    out = train_jit(*swept_args)

    # our final_eval_metric returns max_num_steps.
    # we can filter that down by the max episode length amongst the runs.
    final_eval = out['final_eval_metric']

    # the +1 at the end is to include the done step
    largest_episode = final_eval['returned_episode'].argmax(axis=-2).max() + 1

    def get_first_n_filter(x):
        return x[..., :largest_episode, :]
    out['final_eval_metric'] = jax.tree.map(get_first_n_filter, final_eval)

    if not args.save_runner_state:
        del out['runner_state']

    results_path = get_results_path(args, return_npy=False)  # returns a results directory

    all_results = {
        'argument_order': train_args,
        'out': out,
        'config': config,
        'args': args.as_dict()
    }

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}")
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)

    print("Done.")
