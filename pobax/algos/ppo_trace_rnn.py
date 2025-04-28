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
from gymnax.environments import environment, spaces
from jax._src.nn.initializers import orthogonal, constant

from pobax.config import PPOHyperparams
from pobax.envs import get_env
from pobax.envs.wrappers.gymnax import LogEnvState
from pobax.envs.wrappers.trace import TraceFeatureState
from pobax.models import get_gymnax_network_fn, SimpleNN, ScannedRNN, FullImageCNN, SmallImageCNN
from pobax.models.value import Critic
from pobax.models.discrete import DiscreteActor
from pobax.utils.file_system import get_results_path, numpyify

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value_e: jnp.ndarray
    value_i: jnp.ndarray
    reward_e: jnp.ndarray
    reward_i: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    obs: jnp.ndarray
    next_obs: jnp.ndarray
    traces: jnp.ndarray
    info: jnp.ndarray

class RNDNetwork(nn.Module):
    hidden_size: int
    
    @nn.compact
    def __call__(self, hidden, traces):
        # Traces have shape (batch_size, num_envs, num_features, num_trace_lambdas)
        traces = traces.reshape(*traces.shape[:2], -1)
        # Reshape to (batch_size, num_envs, num_features * num_trace_lambdas)
        embedding = RNDNN(hidden_size=self.hidden_size)(traces)

        return hidden, embedding

class RNDCNNNetwork(nn.Module):
    hidden_size: int
    
    @nn.compact
    def __call__(self, hidden, traces):
        # Traces have shape (batch_size, num_envs, H, W, C, num_trace_lambdas)
        traces = traces.reshape(*traces.shape[:-2], -1)
        # Reshape to (batch_size, num_envs, H, W, C * num_trace_lambdas)
        if traces.shape[-2] >= 20:
            embedding = FullImageCNN(hidden_size=self.hidden_size)(traces)
        else:
            embedding = SmallImageCNN(hidden_size=self.hidden_size)(traces)
        embedding = nn.relu(embedding)
        embedding = RNDNN(hidden_size=self.hidden_size)(embedding)

        return hidden, embedding

class RNDNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        out = nn.Dense(self.hidden_size)(x)
        out = nn.relu(out)
        out = nn.Dense(self.hidden_size)(out)
        out = nn.relu(out)
        out = nn.Dense(self.hidden_size)(out)
        out = nn.relu(out)
        out = nn.Dense(self.hidden_size)(out)
        return out

class TraceRNN(nn.Module):
    lambdas: jnp.ndarray

    @staticmethod
    def initialize_carry(features: jnp.ndarray,
                         n_lambda:   int,
                         dtype=jnp.float32):
        return features[..., None].repeat(n_lambda, axis=-1)

    @functools.partial(
        nn.scan,
        variable_broadcast="params",
        in_axes=0, 
        out_axes=0,
        split_rngs={"params": False},
    )
    @nn.compact
    def __call__(self,
                 carry: jnp.ndarray,
                 x: Tuple[jnp.ndarray, jnp.ndarray]
                 ) -> Tuple[jnp.ndarray, jnp.ndarray]:
        trace = carry
        features, resets = x

        trace = jnp.where(
            resets[:, np.newaxis],
            self.initialize_carry(features, len(self.lambdas)),
            trace,
        )

        leading_dims = (1,) * len(features.shape)
        lambdas = jnp.broadcast_to(self.lambdas, leading_dims + self.lambdas.shape)

        curr_feature_lambda = (1 - resets) * (1 - lambdas) + resets

        next_trace = (1 - resets) * lambdas * trace + curr_feature_lambda * features[..., None]

        unflatten_dims = next_trace.shape[:-2]
        flatten_next_trace = next_trace.reshape((*unflatten_dims, -1))
        return next_trace, flatten_next_trace

class ActorCriticCNNRND(nn.Module):
    action_dim: int
    hidden_size: int = 128
    memoryless: bool = False
    lambdas: jnp.ndarray = None
    traces_in_obs: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        if obs.shape[-2] >= 20:
            embedding = FullImageCNN(hidden_size=self.hidden_size)(obs)
        else:
            embedding = SmallImageCNN(hidden_size=self.hidden_size)(obs)
        embedding = nn.relu(embedding)

        if self.memoryless:
            embedding = SimpleNN(hidden_size=self.hidden_size)(embedding)
        else:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)
        
        if lambdas is not None:
            trace_in = (embedding, dones)
            next_trace, flatten_next_trace = TraceRNN(self.lambdas)(trace, trace_in)
            if self.traces_in_obs:
                embedding = flatten_next_trace

        actor = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)
        pi = actor(embedding)

        critic_e = Critic(hidden_size=self.hidden_size)
        critic_i = Critic(hidden_size=self.hidden_size)

        v_e = critic_e(embedding)
        v_i = critic_i(embedding)

        return hidden, pi, jnp.squeeze(v_e, axis=-1), jnp.squeeze(v_i, axis=-1)

class ActorCriticRND(nn.Module):
    action_dim: int
    hidden_size: int = 128
    memoryless: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        if self.memoryless:
            embedding = SimpleNN(hidden_size=self.hidden_size)(obs)
        else:
            embedding = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
            embedding = nn.relu(embedding)
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)

        actor = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)
        pi = actor(embedding)

        critic_e = Critic(hidden_size=self.hidden_size)
        critic_i = Critic(hidden_size=self.hidden_size)

        v_e = critic_e(embedding)
        v_i = critic_i(embedding)

        return hidden, pi, jnp.squeeze(v_e, axis=-1), jnp.squeeze(v_i, axis=-1)

class PPORND:
    def __init__(self, network,
                 rnd_random_network,
                 rnd_distillation_network,
                 rnd_gae_coeff,
                 rnd_loss_coeff,
                 double_critic: bool = False,
                 ld_weight: float = 0.,
                 alpha: float = 0.,
                 vf_coeff: float = 0.,
                 entropy_coeff: float = 0.01,
                 clip_eps: float = 0.2):
        self.network = network
        self.rnd_random_network = rnd_random_network
        self.rnd_distillation_network = rnd_distillation_network
        self.rnd_gae_coeff = rnd_gae_coeff
        self.rnd_random_network_params = None
        self.rnd_loss_coeff = rnd_loss_coeff
        self.double_critic = double_critic
        self.ld_weight = ld_weight
        self.alpha = alpha
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_eps = clip_eps
        self.act = jax.jit(self.act)
        self.loss = jax.jit(self.loss)

    def act(self, rng: chex.PRNGKey,
            train_state: flax.training.train_state.TrainState,
            hidden_state: chex.Array,
            obs: chex.Array, done: chex.Array):

        # SELECT ACTION
        ac_in = (obs[np.newaxis, :], done[np.newaxis, :])
        hstate, pi, value_e, value_i = self.network.apply(train_state.params, hidden_state, ac_in)
        action = pi.sample(seed=rng)
        log_prob = pi.log_prob(action)
        value_e, value_i, action, log_prob = (
            value_e.squeeze(0),
            value_i.squeeze(0),
            action.squeeze(0),
            log_prob.squeeze(0),
        )
        return value_e, value_i, action, log_prob, hstate
    
    def rnd_act(self, rnd_state: flax.training.train_state.TrainState, rnd_random_hstate: chex.Array, rnd_distillation_hstate: chex.Array,
                traces: chex.Array, done: chex.Array, rnd_reward_coeff):
        traces = traces[np.newaxis, :]
        rnd_random_hstate, random_pred = self.rnd_random_network.apply(self.rnd_random_network_params, rnd_random_hstate, traces)
        rnd_distillation_hstate, distill_pred = self.rnd_distillation_network.apply(rnd_state.params, rnd_distillation_hstate, traces)
        error = (random_pred - distill_pred) * (1 - done[np.newaxis, :, None])
        mse = jnp.square(error).mean(axis=-1)
        reward_i = mse * rnd_reward_coeff
        reward_i = reward_i.squeeze(0)
        # jax.debug.print("reward_i={t}", t=reward_i)
        return rnd_random_hstate, rnd_distillation_hstate, random_pred, distill_pred, error, reward_i

    def loss(self, params, init_hstate, traj_batch, gae_e, targets_e, gae_i, targets_i):
        # RERUN NETWORK
        _, pi, value_e, value_i = self.network.apply(
            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
        )
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE EXTRINSIC VALUE LOSS
        value_pred_clipped_e = traj_batch.value_e + (
                value_e - traj_batch.value_e
        ).clip(-self.clip_eps, self.clip_eps)
        value_losses_e = jnp.square(value_e - targets_e)
        value_losses_clipped_e = jnp.square(value_pred_clipped_e - targets_e)
        value_loss_e = (
            jnp.maximum(value_losses_e, value_losses_clipped_e).mean()
        )

        # CALCULATE INTRINSIC VALUE LOSS
        value_pred_clipped_i = traj_batch.value_i + (
                value_i - traj_batch.value_i
        ).clip(-self.clip_eps, self.clip_eps)
        value_losses_i = jnp.square(value_i - targets_i)
        value_losses_clipped_i = jnp.square(value_pred_clipped_i - targets_i)
        value_loss_i = (
            jnp.maximum(value_losses_i, value_losses_clipped_i).mean()
        )

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)

        # which advantage do we use to update our policy?
        gae = gae_e + self.rnd_gae_coeff * gae_i
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
        value_loss = value_loss_e + value_loss_i

        total_loss = (
                loss_actor
                + self.vf_coeff * value_loss
                - self.entropy_coeff * entropy
        )
        return total_loss, (value_loss_e, value_loss_i, loss_actor, entropy)
    
    def rnd_loss(self, rnd_distillation_params, init_rnd_random_hstate, init_rnd_distillation_hstate, traj_batch):
        random_network_hstate, random_network_out = self.rnd_random_network.apply(self.rnd_random_network_params, init_rnd_random_hstate[0], traj_batch.traces)

        distillation_network_hstate, distillation_network_out = self.rnd_distillation_network.apply(rnd_distillation_params, init_rnd_distillation_hstate[0], traj_batch.traces)

        error = (random_network_out - distillation_network_out) * (
                                1 - traj_batch.done[:, :, None]
                            )
        return jnp.square(error).mean() * self.rnd_loss_coeff

def get_traces(env_state):
    if isinstance(env_state, TraceFeatureState):
        return env_state.trace_features
    else:
        return get_traces(env_state.env_state)

def get_rnd_network_fn(env: environment.Environment, env_params: environment.EnvParams):
    obs_space_shape = env.observation_space(env_params).shape
    if len(obs_space_shape) > 1:
        # assert jnp.all(jnp.array(obs_space_shape[:-1]) == 5)
        network_fn = RNDCNNNetwork
    else:
        network_fn = RNDNetwork
    return network_fn

def get_network_fn(env: environment.Environment, env_params: environment.EnvParams):
    obs_space_shape = env.observation_space(env_params).shape

    if len(obs_space_shape) > 1:
        # assert jnp.all(jnp.array(obs_space_shape[:-1]) == 5)
        network_fn = ActorCriticCNNRND
    else:
        network_fn = ActorCriticRND

    return network_fn

def env_step(runner_state, unused, agent: PPORND, env, env_params, rnd_reward_coeff):
    train_state, rnd_state, env_state, last_obs, last_done, hstate, rnd_random_hstate, rnd_distillation_hstate, rng = runner_state
    rng, _rng = jax.random.split(rng)
    value_e, value_i, action, log_prob, hstate = agent.act(_rng, train_state, hstate, last_obs, last_done)

    last_traces = get_traces(env_state)
    # jax.debug.print("trace_max={t}", t=last_traces)
    rnd_random_hstate, rnd_distillation_hstate, random_pred, distill_pred, error, reward_i = agent.rnd_act(rnd_state, rnd_random_hstate, rnd_distillation_hstate, last_traces, last_done, rnd_reward_coeff)
    
    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, hstate.shape[0])
    obsv, env_state, reward_e, done, info = env.step(rng_step, env_state, action, env_params)
    reward = reward_i + reward_e
    transition = Transition(
        last_done, action, value_e, value_i, reward_e, reward_i, reward, log_prob, last_obs, obsv, last_traces, info
    )

    runner_state = (train_state, rnd_state, env_state, obsv, done, hstate, rnd_random_hstate, rnd_distillation_hstate, rng)
    return runner_state, transition


def calculate_gae(traj_batch, last_val, last_done, gae_lambda, gamma, is_extrinsic):
    def _get_advantages(carry, transition):
        gae, next_value, gae_lambda, next_done, is_extrinsic = carry
        done, value, reward = (
            transition.done,
            jax.lax.select(
                is_extrinsic, transition.value_e, transition.value_i
            ),
            jax.lax.select(
                is_extrinsic, transition.reward_e, transition.reward_i
            ),
        )
        delta = reward + gamma * next_value * (1 - next_done) - value
        gae = delta + gamma * gae_lambda * (1 - next_done) * gae
        return (gae, value, gae_lambda, done, is_extrinsic), gae

    _, advantages = jax.lax.scan(_get_advantages,
                                 (jnp.zeros_like(last_val), last_val, gae_lambda, last_done, is_extrinsic),
                                 traj_batch, reverse=True, unroll=16)
    target = advantages +  jax.lax.select(
                    is_extrinsic, traj_batch.value_e, traj_batch.value_i
                )
    return advantages, target


def filter_period_first_dim(x, n: int):
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        return x[::n]


def make_train(args: PPOHyperparams, rand_key: jax.random.PRNGKey):
    num_updates = (
            args.total_steps // args.num_steps // args.num_envs
    )
    args.minibatch_size = (
            args.num_envs * args.num_steps // args.num_minibatches
    )
    env_key, rand_key = jax.random.split(rand_key)
    env, env_params = get_env(args.env, env_key, args.num_envs,
                                     gamma=args.gamma,
                                     normalize_env=args.normalize_env,
                                     perfect_memory=args.perfect_memory,
                                     action_concat=args.action_concat,
                                     trace_in_obs=args.trace_in_obs,
                                     trace_lambdas=args.trace_lambdas)

    if hasattr(env, 'gamma'):
        args.gamma = env.gamma
    
    double_critic = args.double_critic
    memoryless = args.memoryless

    assert hasattr(env_params, 'max_steps_in_episode')

    # action size
    if isinstance(env.action_space(env_params), spaces.Discrete):
        action_size = env.action_space(env_params).n
    else:
        action_size = env.action_space(env_params).shape[0]
    
    # initialize network
    ac_network_fn = get_network_fn(env, env_params)
    network = ac_network_fn(action_size,
                         hidden_size=args.hidden_size,
                         memoryless=memoryless)
    
    rnd_network_fn = get_rnd_network_fn(env, env_params)
    rnd_random_network = rnd_network_fn(args.rnd_hidden_size)

    rnd_distillation_network = rnd_network_fn(args.rnd_hidden_size)

    steps_filter = partial(filter_period_first_dim, n=args.steps_log_freq)
    update_filter = partial(filter_period_first_dim, n=args.update_log_freq)

    _calculate_gae = calculate_gae

    def train(vf_coeff, ld_weight, alpha, lambda1, lambda0, lr, rnd_lr, rnd_reward_coeff, rng):
        agent = PPORND(network, rnd_random_network, rnd_distillation_network, args.rnd_gae_coeff, args.rnd_loss_coeff, double_critic=double_critic, ld_weight=ld_weight, alpha=alpha, vf_coeff=vf_coeff,
                    clip_eps=args.clip_eps, entropy_coeff=args.entropy_coeff)

        gae_lambda = jnp.array(lambda0)

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

        # initialize rnd network
        init_traces = jnp.ones((1, args.num_envs, *env.observation_space(env_params).shape, args.trace_lambdas.shape[0]))
        init_rnd_random_hstate = ScannedRNN.initialize_carry(args.num_envs, args.rnd_hidden_size)
        rng, _rng = jax.random.split(rng)
        rnd_random_network_params = agent.rnd_random_network.init(_rng, init_rnd_random_hstate, init_traces)
        agent.rnd_random_network_params = rnd_random_network_params
        init_rnd_distillation_hstate = ScannedRNN.initialize_carry(args.num_envs, args.rnd_hidden_size)
        rng, _rng = jax.random.split(rng)
        rnd_distillation_network_params = agent.rnd_distillation_network.init(_rng, init_rnd_distillation_hstate, init_traces)

        # initialize functions
        _env_step = partial(env_step, agent=agent, env=env, env_params=env_params, rnd_reward_coeff=rnd_reward_coeff)
        # Number of parameters
        param_count = sum(x.size for x in jax.tree_leaves(network_params))
        print('Network params number:', param_count)

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
        rnd_tx = optax.chain(
                optax.clip_by_global_norm(args.max_grad_norm),
                optax.adam(rnd_lr, eps=1e-5),
            )
        rnd_train_state = TrainState.create(
                apply_fn=rnd_distillation_network.apply,
                params=rnd_distillation_network_params,
                tx=rnd_tx,
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
            init_init_rnd_random_hstate= ScannedRNN.initialize_carry(args.num_envs, args.rnd_hidden_size)
            init_init_rnd_distillation_hstate = ScannedRNN.initialize_carry(args.num_envs, args.rnd_hidden_size)

            init_runner_state = (
                train_state,
                rnd_train_state,
                env_state,
                init_obsv,
                jnp.zeros(args.num_envs, dtype=bool),
                init_init_hstate,
                init_init_rnd_random_hstate,
                init_init_rnd_distillation_hstate,
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
            env_state = recursive_replace(env_state, starting_runner_state[2], replace_field_names)

        # TRAIN LOOP
        def _update_step(runner_state, i):
            # COLLECT TRAJECTORIES
            initial_hstate = runner_state[-4]
            runner_state, traj_batch = jax.lax.scan(
                _env_step, runner_state, jnp.arange(args.num_steps), args.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, rnd_state, env_state, last_obs, last_done, hstate, rnd_random_hstate, rnd_distillation_hstate, rng = runner_state
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val_e, last_val_i = network.apply(train_state.params, hstate, ac_in)
            last_val_e = last_val_e.squeeze(0)
            last_val_i = last_val_i.squeeze(0)

            advantages_e, targets_e = _calculate_gae(traj_batch, last_val_e, last_done, gae_lambda, args.gamma, True)
            advantages_i, targets_i = _calculate_gae(traj_batch, last_val_i, last_done, gae_lambda, args.gamma, False)

            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                    init_hstate, traj_batch, advantages_e, targets_e, advantages_i, targets_i = batch_info

                    grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, init_hstate, traj_batch, advantages_e, targets_e, advantages_i, targets_i
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                (
                    train_state,
                    init_hstate,
                    traj_batch,
                    advantages_e,
                    targets_e,
                    advantages_i,
                    targets_i,
                    rng,
                ) = update_state

                # SHUFFLE COLLECTED BATCH
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, args.num_envs)
                batch = (init_hstate, traj_batch, advantages_e, targets_e, advantages_i, targets_i)

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
                    advantages_e,
                    targets_e,
                    advantages_i,
                    targets_i,
                    rng,
                )
                return update_state, total_loss

            init_hstate = initial_hstate[None, :]  # TBH
            update_state = (
                train_state,
                init_hstate,
                traj_batch,
                advantages_e,
                targets_e,
                advantages_i,
                targets_i,
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
            
            initial_rnd_random_hstate = runner_state[-3]
            initial_rnd_distillation_hstate = runner_state[-2]
            # UPDATE EXPLORATION STATE
            def _update_ex_epoch(update_state, unused):
                def _update_ex_minbatch(rnd_state, batch_info):
                    init_rnd_random_hstate, init_rnd_distillation_hstate, traj_batch = batch_info
                    rnd_grad_fn = jax.value_and_grad(agent.rnd_loss, has_aux=False)
                    rnd_loss, rnd_grad = rnd_grad_fn(
                            rnd_state.params, init_rnd_random_hstate, init_rnd_distillation_hstate, traj_batch
                    )
                    rnd_state = rnd_state.apply_gradients(grads=rnd_grad)

                    losses = (rnd_loss,)
                    return rnd_state, losses

                (rnd_state, init_rnd_random_hstate, init_rnd_distillation_hstate, traj_batch, rng) = update_state
                # SHUFFLE COLLECTED BATCH
                rng, _rng = jax.random.split(rng)
                permutation = jax.random.permutation(_rng, args.num_envs)
                batch = (init_rnd_random_hstate, init_rnd_distillation_hstate, traj_batch)
                batch_size = args.minibatch_size * args.num_minibatches
                assert (
                    batch_size == args.num_steps * args.num_envs
                ), "batch size must be equal to number of steps * number of envs"
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
                rnd_state, losses = jax.lax.scan(
                    _update_ex_minbatch, rnd_state, minibatches
                )
                update_state = (rnd_state, init_rnd_random_hstate, init_rnd_distillation_hstate, traj_batch, rng)
                return update_state, losses
            
            init_rnd_random_hstate = initial_rnd_random_hstate[None, :]
            init_rnd_distillation_hstate = initial_rnd_distillation_hstate[None, :]
            ex_update_state = (
                rnd_state,
                init_rnd_random_hstate, 
                init_rnd_distillation_hstate,
                traj_batch, 
                rng)
            ex_update_state, ex_loss = jax.lax.scan(
                _update_ex_epoch,
                ex_update_state,
                None,
                args.exploration_update_epochs,
            )
            metric["rnd_loss"] = ex_loss[0].mean() # This might be wrong

            rnd_state = ex_update_state[0]
            rng = ex_update_state[-1]

            runner_state = (train_state, rnd_state, env_state, last_obs, last_done, hstate, rnd_random_hstate, rnd_distillation_hstate, rng)

            return runner_state, metric

        rng, _rng = jax.random.split(rng)
        runner_state = (
            train_state,
            rnd_train_state,
            env_state,
            obsv,
            jnp.zeros((args.num_envs), dtype=bool),
            init_hstate,
            init_rnd_random_hstate,
            init_rnd_distillation_hstate,
            _rng,
        )

        # returned metric has an extra dimension.
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(num_updates), num_updates
        )

        # save metrics only every update_log_freq
        metric = jax.tree.map(update_filter, metric)

        # TODO: offline eval here.
        res = {"runner_state": runner_state, "metric": metric}
        # res = {"runner_state": runner_state, "metric": metric, 'final_eval_metric': eval_traj_batch.info}

        return res

    return train



if __name__ == "__main__":
    # jax.disable_jit(True)
    # okay some weirdness here. NUM_ENVS needs to match with NUM_MINIBATCHES
    args = PPOHyperparams().parse_args()
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