from functools import reduce
from typing import NamedTuple

from collections import deque
from dataclasses import replace
from functools import partial
import inspect
from time import time
# import os
# os.environ["CRAFTAX_RELOAD_TEXTURES"] = "True"

import chex
import flax.linen as nn
import flax.training.train_state
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint

from pobax.algos.ppo import PPO
from pobax.config import GDPPOHyperparams
from pobax.envs import get_env
from pobax.envs.wrappers.gymnax import LogEnvState
from pobax.envs.jax.battleship import Battleship
from pobax.models import ScannedRNN
from pobax.models.actor_critic import CumulantNetwork, HangmanNetwork, ActorCritic, BattleShipActorCritic
from pobax.utils.file_system import get_results_path, numpyify
from pobax.utils.sweep import get_grid_hparams, get_randomly_sampled_hparams


class Transition(NamedTuple):
    done: jnp.ndarray
    hangman: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    cumulant_value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    obs_encoding: jnp.ndarray
    hstate: jnp.ndarray
    cumulant: jnp.ndarray = None
    info: jnp.ndarray = None


class GDTrainState(TrainState):
    cumulant_params: dict
    hangman_params: dict


class GDPPO(PPO):
    def __init__(self, network,
                 hangman_network,
                 double_critic: bool = False,
                 ld_weight: float = 0.,
                 vf_coeff: float = 0.,
                 cumulant_loss_weight: float = 0.5,
                 entropy_coeff: float = 0.01,
                 clip_eps: float = 0.2):
        super().__init__(network,
                         double_critic=double_critic,
                         ld_weight=ld_weight,
                         vf_coeff=vf_coeff,
                         ld_exploration_bonus_scale=0.,
                         entropy_coeff=entropy_coeff,
                         clip_eps=clip_eps)

        self.hangman_network = hangman_network
        self.cumulant_loss_weight = cumulant_loss_weight

    def act(self, rng: chex.PRNGKey,
            train_state: flax.training.train_state.TrainState,
            hidden_state: chex.Array,
            obs: chex.Array, done: chex.Array):

        # SELECT ACTION
        ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
        next_hstate, pi, value, c_value, obs_encoding = self.network.apply(train_state.params, hidden_state, ac_in)
        if c_value is not None:
            c_value = c_value.squeeze(0)
        hangman = self.hangman_network.apply(train_state.hangman_params, obs)

        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        value, action, log_prob, obs_encoding = (
            value.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
            obs_encoding.squeeze(0)
        )
        return value, c_value, action, log_prob, next_hstate, hangman, obs_encoding

    def loss(self, params, init_hstate, traj_batch, gae, targets, cumulant_targets, next_vals):
        # RERUN NETWORK
        _, pi, value, cumulant_value, obs_encoding = self.network.apply(
            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
        )
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
                value - traj_batch.value
        ).clip(-self.clip_eps, self.clip_eps)
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            jnp.maximum(value_losses, value_losses_clipped).mean()
        )

        general_discrep_loss = 0
        if cumulant_targets is not None:
            value_loss += self.cumulant_loss_weight * jnp.square(cumulant_value - cumulant_targets).mean()

            # Lambda discrepancy loss
            if self.double_critic:
                # value_loss = self.ld_weight * (jnp.square(value[..., 0] - value[..., 1])).mean() + \
                #              (1 - self.ld_weight) * value_loss
                general_discrep_loss = (jnp.square(cumulant_value[..., 0, :] - cumulant_value[..., 1, :]).mean())

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)

        # which advantage do we use to update our policy?
        # if self.double_critic:
        #     gae = (self.alpha * gae[..., 0] +
        #            (1 - self.alpha) * gae[..., 1])
        #
        #     ld_exploration_bonus = jnp.abs(next_vals[..., 0] - next_vals[..., 1])
        #
        #     gae += self.ld_exploration_bonus_scale * ld_exploration_bonus

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
                + self.ld_weight * general_discrep_loss
                - self.entropy_coeff * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)


def env_step(runner_state, unused,
             agent: GDPPO, env, env_params):
    train_state, env_state, last_obs, last_done, last_hstate, rng = runner_state
    rng, _rng = jax.random.split(rng)

    # gamma_offset is between -1 and 1
    value, cumulant_value, action, log_prob, hstate, hangman, last_obs_encoding = agent.act(_rng, train_state, last_hstate, last_obs, last_done)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, hstate.shape[0])
    obsv, env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)

    transition = Transition(
        last_done, hangman, action, value, cumulant_value, reward, log_prob, last_obs, last_obs_encoding, last_hstate, None, info
    )

    runner_state = (train_state, env_state, obsv, done, hstate, rng)
    return runner_state, transition


def calculate_gae(traj_batch, last_val, last_done, gae_lambda, gamma):
    def _get_advantages(carry, transition):
        gae, next_value, gae_lambda, next_done = carry
        done, value, reward = transition.done, transition.value, transition.reward
        delta = reward + gamma * next_value * (1 - next_done) - value
        gae = delta + gamma * gae_lambda * (1 - next_done) * gae
        return (gae, value, gae_lambda, done), gae

    _, advantages = jax.lax.scan(_get_advantages,
                                 (jnp.zeros_like(last_val), last_val, gae_lambda, last_done),
                                 traj_batch, reverse=True, unroll=16)
    target = advantages + traj_batch.value
    return advantages, target


def calculate_gvf_lambda(traj_batch, last_cumulant_val, last_done, gae_lambda):
    def _get_advantages(carry, transition):
        gae_cumulant, next_cumulant_value, gae_lambda, next_done = carry
        done, cumulant_value, reward, cumulant, hangman, obs\
            = (transition.done, transition.cumulant_value, transition.reward,
               transition.cumulant, transition.hangman,
               transition.obs)
        next_done = next_done[..., None]

        delta_cumulant = cumulant + hangman * next_cumulant_value * (1 - next_done) - cumulant_value
        gae_cumulant = delta_cumulant + hangman * gae_lambda * (1 - next_done) * gae_cumulant
        return (gae_cumulant, cumulant_value, gae_lambda, done), gae_cumulant

    _, adv_cumulant = jax.lax.scan(_get_advantages,
                                   (jnp.zeros_like(last_cumulant_val), last_cumulant_val, gae_lambda, last_done),
                                   traj_batch, reverse=True, unroll=16)
    gvf_target = adv_cumulant + traj_batch.cumulant_value
    return gvf_target


def filter_period_first_dim(x, n: int):
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        return x[::n]


def make_train(args: GDPPOHyperparams, rand_key: jax.random.PRNGKey):
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
                              action_concat=args.action_concat,
                              reward_concat=args.reward_concat)

    if hasattr(env, 'gamma'):
        args.gamma = env.gamma

    assert hasattr(env_params, 'max_steps_in_episode')

    cumulant_size = None
    if args.cumulant_transform == 'random_proj':
        # randomly projected hidden state
        cumulant_size = args.cumulant_map_size
    else:
        if args.cumulant_type == 'rew':
            cumulant_size = 1
        elif args.cumulant_type == 'hs':
            # Raw hidden state and raw hs difference
            cumulant_size = args.hidden_size
        elif args.cumulant_type == 'obs':
            obs_shape = env.observation_space(env_params).shape
            if len(obs_shape) > 1:
                cumulant_size = reduce(lambda x, y: x * y, obs_shape)
            else:
                cumulant_size = obs_shape[0]
        elif args.cumulant_type == 'enc_obs':
            cumulant_size = args.hidden_size

    if args.add_reward_to_cumulant:
        cumulant_size += 1

    if isinstance(env, Battleship) or ((hasattr(env, '_unwrapped') and isinstance(env._unwrapped, Battleship))):
        network = BattleShipActorCritic(env.action_space(env_params),
                                        memoryless=args.memoryless,
                                        double_critic=args.double_critic,
                                        hidden_size=args.hidden_size,
                                        cumulant_size=cumulant_size)
    else:
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


    steps_filter = partial(filter_period_first_dim, n=args.steps_log_freq)
    update_filter = partial(filter_period_first_dim, n=args.update_log_freq)

    # Used for vmapping over our double critic.
    transition_axes_map = Transition(
        None, None, None, None, 2, None, None, None, None, None
    )

    _calculate_gae = calculate_gae
    _calculate_gvf_lambda = calculate_gvf_lambda

    if args.double_critic:
        # last_val is index 1 here b/c we squeezed earlier.
        # _calculate_gae = jax.vmap(calculate_gae,
        #                          in_axes=[transition_axes_map, 1, None, 0, None],
        #                          out_axes=2)

        _calculate_gvf_lambda = jax.vmap(_calculate_gvf_lambda,
                                         in_axes=[transition_axes_map, 1, None, 0],
                                         out_axes=2)

    def train(sweep_args_dict, rng):
        lr, ld_weight, vf_coeff, lambda0, lambda1, entropy_coeff, cumulant_loss_weight = \
            sweep_args_dict['lr'], sweep_args_dict['ld_weight'], sweep_args_dict['vf_coeff'], \
                sweep_args_dict['lambda0'], sweep_args_dict['lambda1'], sweep_args_dict['entropy_coeff'], \
                sweep_args_dict['cumulant_loss_weight']

        agent = GDPPO(network, hangman_network,
                      double_critic=args.double_critic, ld_weight=ld_weight, vf_coeff=vf_coeff,
                      clip_eps=args.clip_eps, entropy_coeff=entropy_coeff,
                      cumulant_loss_weight=cumulant_loss_weight)

        # initialize functions
        _env_step = partial(env_step, agent=agent, env=env, env_params=env_params)

        gae_lambda = jnp.array(lambda0)

        def linear_schedule(count):
            frac = (
                    1.0
                    - (count // (args.num_minibatches * args.update_epochs))
                    / num_updates
            )
            return lr * frac


        # INIT NETWORK
        rng, network_rng, hangman_rng, cumulant_rng = jax.random.split(rng, 4)
        init_x = (
            jnp.zeros(
                (1, args.num_envs, *env.observation_space(env_params).shape)
            ),
            jnp.zeros((1, args.num_envs)),
        )
        init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
        network_params = agent.network.init(network_rng, init_hstate, init_x)

        hangman_params = agent.hangman_network.init(hangman_rng, init_x[0])
        cumulant_params = None
        if args.cumulant_transform == 'random_proj':
            if args.cumulant_type == 'obs':
                inp = init_x[0]
            elif args.cumulant_type == 'enc_obs':
                inp = jnp.zeros((1, args.num_envs, args.hidden_size))
            elif args.cumulant_type == 'hs':
                inp = init_hstate
            cumulant_params = cumulant_network.init(cumulant_rng, inp)

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

        train_state = GDTrainState.create(
            apply_fn=agent.network.apply,
            params=network_params,
            tx=tx,
            cumulant_params=cumulant_params,
            hangman_params=hangman_params,
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
            initial_hstate = runner_state[4]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(args.num_steps), args.num_steps
            )

            train_state, env_state, last_obs, last_done, hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val, last_cumulant_val, last_encoded_obs = network.apply(train_state.params, hstate, ac_in)

            # CALCULATE CUMULANTS
            cs, last_c = None, None
            if args.cumulant_type == 'obs':
                cs = traj_batch.obs
                if len(last_obs.shape[2:]) > 1:
                    cs = cs.reshape((*cs.shape[:2], -1))
                    last_c = last_obs.reshape((last_obs.shape[0], -1))
                else:
                    last_c = last_obs

            elif args.cumulant_type == 'hs':
                cs, last_c = traj_batch.hstate, hstate
            elif args.cumulant_type == 'enc_obs':
                cs = traj_batch.obs_encoding
                last_c = last_encoded_obs

            cumulant = cs
            if args.cumulant_diff:
                next_cs = jnp.concatenate((cs[1:], last_c[None, ...]), axis=0)
                cumulant = (1 - traj_batch.done[..., None]) * next_cs - cs

            if args.cumulant_transform == 'random_proj':
                cumulant = cumulant_network.apply(train_state.cumulant_params, cumulant)

            if args.scale_cumulant:
                cumulant = (1 - args.gamma) * cumulant

            if args.add_reward_to_cumulant:
                cumulant = jnp.concatenate([cumulant, traj_batch.reward[..., None]], axis=-1)

            traj_batch = traj_batch._replace(cumulant=cumulant)

            # CALCULATE ADVANTAGE
            next_vals = jnp.concatenate((traj_batch.value[1:], last_val), axis=0)
            last_val = last_val.squeeze(0)
            last_cumulant_val = last_cumulant_val.squeeze(0)

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done, gae_lambda, args.gamma)
            # CALCULATE LD + GVF TARGETS
            cumulant_gae_lambda = lambda0
            if args.double_critic:
                cumulant_gae_lambda = jnp.array([lambda0, lambda1])
            cumulant_targets = _calculate_gvf_lambda(traj_batch, last_cumulant_val, last_done, cumulant_gae_lambda)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages, targets, cumulant_targets, next_vals = batch_info

                    grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages,
                        targets, cumulant_targets, next_vals
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages,
                    targets,
                    cumulant_targets,
                    next_vals,
                    rng,
                ) = update_state

                # SHUFFLE COLLECTED BATCH
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, args.num_envs)
                batch = (init_hstate, traj_batch, advantages, targets, cumulant_targets, next_vals)

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
                    cumulant_targets,
                    next_vals,
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
                cumulant_targets,
                next_vals,
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
            jnp.zeros(args.num_envs, dtype=bool),
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
            jnp.zeros(args.num_envs, dtype=bool),
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
    # okay some weirdness here. NUM_ENVS needs to match with NUM_MINIBATCHES
    args = GDPPOHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    rng = jax.random.PRNGKey(args.seed)
    make_train_rng, rng = jax.random.split(rng)
    rngs = jax.random.split(rng, args.n_seeds)
    train_fn = make_train(args, make_train_rng)

    # train_args = list(inspect.signature(train_fn).parameters.keys())
    #
    # vmaps_train = train_fn
    # swept_args = deque()
    #
    # # we need to go backwards, since JAX returns indices
    # # in the order in which they're vmapped.
    # for i, arg in reversed(list(enumerate(train_args))):
    #     dims = [None] * len(train_args)
    #     dims[i] = 0
    #     vmaps_train = jax.vmap(vmaps_train, in_axes=dims)
    #     if arg == 'rng':
    #         swept_args.appendleft(rngs)
    #     else:
    #         assert hasattr(args, arg)
    #         swept_args.appendleft(getattr(args, arg))
    #
    # train_jit = jax.jit(vmaps_train)

    if args.sweep_type == 'grid':
        hparams, _ = get_grid_hparams(args)
    elif args.sweep_type == 'random':
        _rng, rng = jax.random.split(rng)
        hparams = get_randomly_sampled_hparams(_rng, args, n_samples=args.n_random_hparams)
    else:
        raise NotImplementedError

    vmap_seeds_train_fn = jax.vmap(train_fn, in_axes=[None, 0])
    vmap_train_fn = jax.vmap(vmap_seeds_train_fn, in_axes=[0, None])
    train_jit = jax.jit(vmap_train_fn)

    t = time()
    out = jax.block_until_ready(train_jit(hparams, rngs))
    new_t = time()
    total_runtime = new_t - t
    print('Total runtime:', total_runtime)

    # our final_eval_metric returns max_num_steps.
    # we can filter that down by the max episode length amongst the runs.
    final_eval = out['final_eval_metric']
    final_train_state = out['runner_state'][0]

    if not args.save_runner_state:
        del out['runner_state']

    results_path = get_results_path(args, return_npy=False)  # returns a results directory

    all_results = {
        'swept_hparams': hparams,
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
