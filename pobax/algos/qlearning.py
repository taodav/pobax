import os
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
from typing import NamedTuple

from collections import deque
from dataclasses import replace
from functools import partial
import inspect

from flax.training.train_state import TrainState
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import jumanji.specs
import numpy as np
import optax
import orbax.checkpoint
import flax.linen as nn

from pobax.config import Hyperparams
from pobax.envs import get_env, jumanji_envs
from pobax.envs.wrappers import LogEnvState
from pobax.models import get_network_fn, ScannedRNN
from pobax.utils.file_system import get_results_path, numpyify_and_save
from pathlib import Path
from jax._src.nn.initializers import orthogonal, constant

class QNetwork(nn.Module):
    n_actions: int
    hidden_size: int = 8
    depth: int = 1

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        features = obs
        for _ in range(self.depth):
            features = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
                features
            )
            features = nn.relu(features)
        q_vals = nn.Dense(self.n_actions, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            features
        )
        return hidden, q_vals, features

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    obs: jnp.ndarray
    features: jnp.ndarray
    info: jnp.ndarray

def eps_greedy_exploration(rng, q_vals):
    rng_a, rng_e = jax.random.split(
        rng
    )  # a key for sampling random actions and one for picking
    greedy_actions = jnp.argmax(q_vals, axis=-1)
    chosed_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape) < 0.1,  # pick the actions that should be random
        jax.random.randint(
            rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
        ),  # sample random actions,
        greedy_actions,
        )
    return chosed_actions


def env_step(runner_state, unused, network, env, env_params):
    train_state, env_state, last_obs, last_done, hstate, rng = runner_state
    rng, rng_a = jax.random.split(rng)

    # SELECT ACTION
    ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
    # network output q_values of shape (action_size,)
    hstate, q_values, features = network.apply(train_state.params, hstate, ac_in)
    value = q_values.squeeze(0)
    features = features.squeeze(0)
    # epsilon-greedy
    _rngs = jax.random.split(rng_a, hstate.shape[0])
    action = jax.vmap(eps_greedy_exploration)(_rngs, value)
    
    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, hstate.shape[0])
    obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)
    transition = Transition(
        last_done, action, value, reward, last_obs, features, info
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
                                     )
    
    if hasattr(env, 'gamma'):
        config['GAMMA'] = env.gamma

    assert hasattr(env_params, 'max_steps_in_episode')

    memoryless = config["MEMORYLESS"]

    action_size = env.action_space(env_params).n
    network_fn = QNetwork
    network = network_fn(action_size, hidden_size=config['HIDDEN_SIZE'], depth = config['DEPTH'])

    steps_filter = partial(filter_period_first_dim, n=config['STEPS_LOG_FREQ'])
    update_filter = partial(filter_period_first_dim, n=config['UPDATE_LOG_FREQ'])

    _env_step = partial(env_step, network=network, env=env, env_params=env_params)

    def train(lr, rng):
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
            train_state, env_state, last_obs, last_done, hstate, rng = runner_state

            def _learn_epoch(update_state, _):
                def _learn_minibatch(update_state, minibatch):
                    init_hstate, minibatch = minibatch
                    def _compute_targets(q_vals, reward, done):
                        # q_vals: [num_steps, batch_size, num_actions]
                        # reward, done: [num_steps, batch_size]

                        # Compute max Q-values for next states
                        max_next_q_vals = jnp.max(q_vals[1:], axis=-1)  # Shape: [num_steps - 1, batch_size]

                        # Compute targets using the standard Q-learning update
                        targets = reward[:-1] + config["GAMMA"] * (1 - done[:-1]) * max_next_q_vals

                        # For terminal steps where done[t] = 1, target is just reward[t]
                        targets = jnp.where(done[:-1], reward[:-1], targets)

                        return targets  # Shape: [num_steps - 1, batch_size]
                    
                    def _loss_fn(params):
                        _, q_vals, _ = network.apply(params, initial_hstate, (minibatch.obs, minibatch.done))
                        target_q_vals = jax.lax.stop_gradient(q_vals)
                        # last_q = target_q_vals[-1].max(axis=-1)

                        compute_targets = _compute_targets
                        target = compute_targets(
                            # last_q,  # q_vals at t=NUM_STEPS-1
                            target_q_vals,
                            minibatch.reward,
                            minibatch.done,
                        ).reshape(
                            -1
                        )  # (num_steps-1*batch_size,)
                        selected_actions = jnp.expand_dims(minibatch.action, axis=-1)

                        chosen_action_qvals = jnp.take_along_axis(
                            q_vals,
                            selected_actions,
                            axis=-1,
                        ).squeeze(axis=-1)  # (num_steps, num_agents, batch_size,)
                        chosen_action_qvals = chosen_action_qvals[:-1].reshape(-1)  # (num_steps-1*batch_size,)

                        loss = 0.5 * jnp.square(chosen_action_qvals - target).mean()

                        return loss, chosen_action_qvals
                    
                    (train_state, rng) = update_state
                    (loss, chosen_action_qvals), grads = jax.value_and_grad(
                        _loss_fn, has_aux=True
                    )(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return (train_state, rng), (loss, chosen_action_qvals)
                    
                (
                    train_state,
                    rng,
                ) = update_state

                # SHUFFLE MINIBATCHES
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, config["NUM_ENVS"])
                batch = (init_hstate, traj_batch)

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
                update_state, (loss, chosen_action_qvals) = jax.lax.scan(
                    _learn_minibatch, (train_state, rng), minibatches
                )
                
                return update_state, (loss, chosen_action_qvals)

            update_state = (
                train_state,
                rng,
            )
            update_state, loss_info = jax.lax.scan(
                _learn_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]

            # save metrics only every steps_log_freq
            metric = traj_batch.info
            metric = jax.tree.map(steps_filter, metric)
            metric.update({"loss": loss_info[0]})
            metric.update({"chosen_action_qvals": loss_info[1]})
            rng = update_state[-1]
            if config.get("SAVE_FEATURES"):
                features = traj_batch.features
                metric.update({"features": features}) 

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

class QlearningHyperparams(Hyperparams):
    env: str = 'fully_observable_4x3'
    memoryless: bool = False
    save_features: bool = False
    save_runner_state: bool = False

    lr: list[float] = [2.5e-3] # learning rate
    vf_coeff: list[float] = [0.5]

    entropy_coeff: float = 0.01
    clip_eps: float = 0.2  # PPO log grad clipping
    max_grad_norm: float = 0.5

    not_anneal_lr: bool = True
    hidden_size: int = 8
    depth: int = 1
    num_minibatches: int = 4
    num_envs: int = 4
    num_steps: int = 128
    update_epochs: int = 4

    def process_args(self) -> None:
        self.vf_coeff = jnp.array(self.vf_coeff)
        self.lr = jnp.array(self.lr)
        self.entropy_coeff = jnp.array(self.entropy_coeff)
        self.clip_eps = jnp.array(self.clip_eps)
        self.max_grad_norm = jnp.array(self.max_grad_norm)

        self.num_updates = self.total_steps // self.num_steps // self.num_envs
        self.minibatch_size = self.num_envs * self.num_steps // self.num_minibatches

if __name__ == "__main__":
    # jax.disable_jit(True)
    # okay some weirdness here. NUM_ENVS needs to match with NUM_MINIBATCHES

    args = QlearningHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    config = {
        "NUM_ENVS": 4,
        "NUM_EVAL_ENVS": 10,
        "NUM_STEPS": args.num_steps,
        "TOTAL_TIMESTEPS": args.total_steps,
        "SAVE_FEATURES": args.save_features,
        "DEFAULT_MAX_STEPS_IN_EPISODE": args.default_max_steps_in_episode,
        "UPDATE_EPOCHS": 4,
        "NUM_MINIBATCHES": 4,
        "GAMMA": 0.99,
        "MEMORYLESS": args.memoryless,
        "CLIP_EPS": 0.2,
        "ENT_COEF": args.entropy_coeff,
        "VF_COEF": args.vf_coeff,
        "MAX_GRAD_NORM": 0.5,
        "HIDDEN_SIZE": args.hidden_size,
        "DEPTH": args.depth,
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
    ##
    # all_results['out']['metric']['features'] should have shape ['num_lr', 'num_update', 'num_steps', 'seeds', 'env', 'num_features']

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}")
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)
    
    # numpyify_and_save(results_path, all_results)

    print("Done.")