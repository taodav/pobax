from functools import partial
from pathlib import Path

import chex
from flax.training.train_state import TrainState
import jax
import jax.numpy as jnp
import numpy as np
from navix import observations
import optax
import orbax.checkpoint
from PIL import Image

from pobax.algos.ppo import PPO, Transition
from pobax.envs import get_env
from pobax.config import PPOHyperparams
from pobax.models import get_gymnax_network_fn, ScannedRNN
from pobax.utils.video import navix_overlay_obs_on_rgb

from definitions import ROOT_DIR


def load_train_state(fpath: Path, key: chex.PRNGKey):
    env_key, key = jax.random.split(key)
    # load our params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(fpath)
    args = restored['args']
    args = PPOHyperparams().from_dict(args)

    env, env_params = get_env(args.env, env_key,
                              gamma=args.gamma,
                              normalize_image=False,
                              action_concat=args.action_concat)

    double_critic = args.double_critic
    memoryless = args.memoryless

    network_fn, action_size = get_gymnax_network_fn(env, env_params, memoryless=memoryless)

    network = network_fn(action_size,
                         double_critic=double_critic,
                         hidden_size=args.hidden_size)

    agent = PPO(network,
                double_critic=double_critic,
                ld_weight=args.ld_weight.item(),
                alpha=args.alpha.item(),
                vf_coeff=args.vf_coeff.item(),
                clip_eps=args.clip_eps, entropy_coeff=args.entropy_coeff)

    ts_dict = jax.tree.map(lambda x: x[0, 0, 0, 0, 0, 0, 0], restored['final_train_state'])
    tx = optax.adam(args.lr)
    ts = TrainState.create(apply_fn=network.apply,
                           params=ts_dict['params'],
                           tx=tx)

    return env, env_params, args, agent, ts


def env_step(runner_state, unused, agent: PPO, env, env_params):
    train_state, prev_env_state, last_obs, last_done, hstate, rng = runner_state
    rng, _rng = jax.random.split(rng)
    value, action, log_prob, hstate = agent.act(_rng, train_state, hstate, last_obs, last_done)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, hstate.shape[0])
    obsv, env_state, reward, done, info = env.step(rng_step, prev_env_state, action, env_params)
    transition = Transition(
        last_done, action, value, reward, log_prob, last_obs, info
    )
    runner_state = (train_state, env_state, obsv, done, hstate, rng)
    return runner_state, (transition, prev_env_state)


if __name__ == "__main__":
    key = jax.random.key(2024)
    n_envs = 2
    n_steps = 1000
    key, load_key = jax.random.split(key)

    # ckpt_path = Path('/Users/ruoyutao/Documents/pobax/results/navix_01_ppo_memoryless/Navix-DMLab-Maze-01-v0_seed(2024)_time(20250211-104715)_072c9acb4cb4a5a026ed32984afa19ad')
    ckpt_path = Path('/Users/ruoyutao/Documents/pobax/results/navix_02_ppo/Navix-DMLab-Maze-02-v0_seed(2024)_time(20250210-182201)_ce243ea47b8d2945f871e39d17c230a3')

    env, env_params, args, agent, ts = load_train_state(ckpt_path, load_key)

    init_x = (
        jnp.zeros(
            (1, args.num_envs, *env.observation_space(env_params).shape)
        ),
        jnp.zeros((1, args.num_envs)),
    )

    init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)
    key, _key = jax.random.split(key)
    init_rng = jax.random.split(_key, args.num_envs)
    init_obsv, env_state = env.reset(init_rng, env_params)

    key, _key = jax.random.split(key)
    runner_state = (
        ts,
        env_state,
        init_obsv,
        jnp.zeros(args.num_envs, dtype=bool),
        init_hstate,
        _key,
    )
    # initialize functions
    _env_step = partial(env_step, agent=agent, env=env, env_params=env_params)

    runner_state, (traj_batch, states) = jax.lax.scan(
        _env_step, runner_state, jnp.arange(n_steps), n_steps
    )

    def numpify_state(leaf):
        if isinstance(leaf, jax._src.prng.PRNGKeyArray):
            leaf = jax.random.key_data(leaf)
        return np.array(leaf[:, 0])
    states = jax.tree.map(numpify_state, states)

    print(f"Collected {n_steps} samples. Turning into MP4 now.")

    if '-F-' not in args.env:
        obs = np.array(traj_batch.obs[:, 0])
        np_images = navix_overlay_obs_on_rgb(obs, states)
    else:
        states = [jax.tree.map(lambda x: jnp.array(x[i]), states.env_state.state) for i in states.env_state.t]
        np_images = [np.array(observations.rgb(s)) for s in states]

    # concat_obs = jnp.concatenate((obs[:, 0], obs[:, 1]), axis=1)
    #
    imgs = [Image.fromarray(img) for img in np_images]
    vod_path = Path(ROOT_DIR, 'results', "navix_maze.gif")
    imgs[0].save(vod_path, save_all=True, append_images=imgs[1:], duration=100, loop=0)

    print(f"saved to {vod_path}")
    print()


