"""
Runs without a jax environment
"""
from functools import partial

import chex
from flax.training.train_state import TrainState
import gymnasium as gym
from gymnasium import ObservationWrapper
from gymnasium.wrappers import PixelObservationWrapper, NormalizeObservation, ResizeObservation
import jax
import jax.numpy as jnp
import numpy as np
import optax
from tqdm import tqdm

from pobax.config import PPOHyperparams
from pobax.models import get_network_fn, ScannedRNN
from pobax.algos.ppo import PPO, calculate_gae, Transition


class PixelOnlyObservationWrapper(PixelObservationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.observation_space['pixels']


def make_train(args: PPOHyperparams, rand_key: chex.PRNGKey):
    wrappers = [
        PixelOnlyObservationWrapper,
        partial(ResizeObservation, shape=64),
        NormalizeObservation
    ]

    env = gym.vector.make(args.env, num_envs=args.num_envs, wrappers=wrappers, render_mode='rgb_array')

    network_fn, obs_shape, action_size = get_network_fn(env, memoryless=args.memoryless)

    network = network_fn(action_size,
                         double_critic=args.double_critic,
                         hidden_size=args.hidden_size)
    agent = PPO(network, args.double_critic, args.ld_weight, args.alpha, args.vf_coeff)

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
            (1, *obs_shape)
        ),
        jnp.zeros((1, obs_shape[0])),
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

    pbar = tqdm(total=args.num_steps)

    # COLLECT MINIBATCH
    last_obs, info = env.reset()
    last_done = jnp.zeros_like(init_x[-1], dtype=bool)
    hstate = ScannedRNN.initialize_carry(last_obs.shape[0], args.hidden_size)

    step = 0
    while step < args.total_steps:

        transitions = []
        for i in range(args.num_steps):
            act_rng, rng = jax.random.split(rng)
            value, action, log_prob, hstate = agent.act(act_rng, train_state, hstate, last_obs, last_done)
            obs, reward, dones, _, info = env.step(action)
            transitions.append(Transition(
                last_done, action, value, reward, log_prob, last_obs, info
            ))
            last_obs, last_done = obs, dones

        # UPDATE FROM MINIBATCH
        @jax.jit
        def _update_step(rng, transitions, last_obs, last_done, hstate, train_state):
            # here we do everything post collecting
            traj_batch = jax.tree.map(lambda *leaves: jnp.stack(leaves, axis=0), *transitions)

            # CALCULATE ADVANTAGE AND TARGETS
            ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
            _, _, last_val = network.apply(train_state.params, hstate, ac_in)
            last_val = last_val.squeeze(0)

            advantages, targets = calculate_gae(traj_batch, last_val, last_done, gae_lambda)

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

        train_state, step_loss = _update_step()

        pbar.update(args.num_steps)

        # maybe calculate episode statistics here?
        # we should have a jitted function that does that here.


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
    args = PPOHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    # Check that we're not trying to vmap over hyperparams
    for k, v in args.as_dict().items():
        if isinstance(v, list):
            assert len(v) == 1, "Can't run with multiple hyperparams."
            setattr(args, k, v[0])

    np.random.seed(args.seed)
    key = jax.random.PRNGKey(args.seed)
    make_train(args, key)

    print()
