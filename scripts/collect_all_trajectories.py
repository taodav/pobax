from functools import partial
from pathlib import Path
from typing import Union, NamedTuple

import chex
import jax
import jax.numpy as jnp
from jax_tqdm import scan_tqdm
import numpy as np
from tap import Tap
from flax.training import orbax_utils
import orbax.checkpoint

from gymnax.environments import environment
from pobax.models.network import ScannedRNN
from pobax.utils.file_system import load_train_state, make_hash_md5


class CollectHyperparams(Tap):
    memoryless_path: Union[str, Path]
    memoryless_LD_path: Union[str, Path]
    memoryless_skip_path: Union[str, Path]
    memoryless_skip_LD_path: Union[str, Path]
    rnn_skip_path: Union[str, Path]
    rnn_skip_LD_path: Union[str, Path]
    rnn_path: Union[str, Path]
    rnn_LD_path: Union[str, Path]
    behavior_path: Union[str, Path]

    update_idx_to_take: int = None

    num_envs: int = 4
    n_samples: int = int(2.5e5)

    seed: int = 2024
    platform: str = 'gpu'

    def configure(self) -> None:
        self.add_argument('--memoryless_path', type=Path)
        self.add_argument('--memoryless_LD_path', type=Path)
        self.add_argument('--memoryless_skip_path', type=Path)
        self.add_argument('--memoryless_skip_LD_path', type=Path)
        self.add_argument('--rnn_skip_path', type=Path)
        self.add_argument('--rnn_skip_LD_path', type=Path)
        self.add_argument('--rnn_path', type=Path)
        self.add_argument('--rnn_LD_path', type=Path)
        self.add_argument('--behavior_path', type=Path)


def ppo_pocman_step(runner_state, unused,
                    behavior_network, memoryless_network, memoryless_LD_network, memoryless_skip_network, memoryless_skip_LD_network,
                    rnn_network, rnn_LD_network, rnn_skip_network, rnn_skip_LD_network,
                    env, env_params):
    def get_state(s):
        if hasattr(s, 'env_state'):
            return get_state(s.env_state)
        else:
            return s

    # (behavior_ts, rnn_ts_0, rnn_ts_1, env_state, last_obs, last_done,
    #     behavior_hstate, rnn_hstate_0, rnn_hstate_1, rng) = runner_state
    (behavior_ts, memoryless_ts, memoryless_LD_ts, memoryless_skip_ts, memoryless_skip_LD_ts, rnn_ts, rnn_LD_ts, rnn_skip_ts, rnn_skip_LD_ts,
     env_state, last_obs, last_done, behavior_hstate, rnn_hstate, rnn_LD_hstate, rng) = runner_state
    rng, _rng = jax.random.split(rng)

    # SELECT ACTION
    ac_in = (last_obs[np.newaxis, :], last_done[np.newaxis, :])
    next_behavior_hstate, pi, value, _ = behavior_network.apply(behavior_ts.params, behavior_hstate, ac_in)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)
    value, action, log_prob = (
        value.squeeze(0),
        action.squeeze(0),
        log_prob.squeeze(0),
    )

    # get our RNN hidden states that we're sampling
    _, _, _, memoryless_embedding = memoryless_network.apply(memoryless_ts.params, behavior_hstate, ac_in)
    _, _, _, memoryless_LD_embedding = memoryless_LD_network.apply(memoryless_LD_ts.params, behavior_hstate, ac_in)
    _, _, _, memoryless_skip_embedding = memoryless_skip_network.apply(memoryless_skip_ts.params, behavior_hstate, ac_in)
    _, _, _, memoryless_skip_LD_embedding = memoryless_skip_LD_network.apply(memoryless_skip_LD_ts.params, behavior_hstate, ac_in)
    next_rnn_hstate, _, _, rnn_embedding = rnn_network.apply(rnn_ts.params, rnn_hstate, ac_in)
    next_rnn_LD_hstate, _, _, rnn_LD_embedding = rnn_LD_network.apply(rnn_LD_ts.params, rnn_LD_hstate, ac_in)
    _, _, _, rnn_skip_embedding = rnn_skip_network.apply(rnn_skip_ts.params, behavior_hstate, ac_in)
    _, _, _, rnn_skip_LD_embedding = rnn_skip_LD_network.apply(rnn_skip_LD_ts.params, behavior_hstate, ac_in)

    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, next_behavior_hstate.shape[0])
    obsv, next_env_state, reward, done, info = env.step(rng_step, env_state, action, env_params)

    # transition = Transition(
    #     last_done, action, value, reward, log_prob, last_obs, info
    # )
    
    datum = {
        'x_memoryless_embedding': memoryless_embedding.squeeze(0),
        'x_memoryless_LD_embedding': memoryless_LD_embedding.squeeze(0),
        'x_memoryless_skip_embedding': memoryless_skip_embedding.squeeze(0),
        'x_memoryless_skip_LD_embedding': memoryless_skip_LD_embedding.squeeze(0),
        'x_rnn_embedding': rnn_embedding.squeeze(0),
        'x_rnn_LD_embedding': rnn_LD_embedding.squeeze(0),
        'x_rnn_skip_embedding': rnn_skip_embedding.squeeze(0),
        'x_rnn_skip_LD_embedding': rnn_skip_LD_embedding.squeeze(0),
        'observation': last_obs,
    }
    runner_state = (behavior_ts, memoryless_ts, memoryless_LD_ts, memoryless_skip_ts, memoryless_skip_LD_ts, rnn_ts, rnn_LD_ts, rnn_skip_ts, rnn_skip_LD_ts,
                    next_env_state, obsv, done, next_behavior_hstate, next_rnn_hstate, next_rnn_LD_hstate, rng)
    return runner_state, datum


def make_collect(args: CollectHyperparams, key: chex.PRNGKey):
    steps_to_collect = args.n_samples // args.num_envs
    args.behavior_path = Path(args.behavior_path).resolve()
    args.memoryless_path = Path(args.memoryless_path).resolve()
    args.memoryless_LD_path = Path(args.memoryless_LD_path).resolve()
    args.memoryless_skip_path = Path(args.memoryless_skip_path).resolve()
    args.memoryless_skip_LD_path = Path(args.memoryless_skip_LD_path).resolve()
    args.rnn_path = Path(args.rnn_path).resolve()
    args.rnn_LD_path = Path(args.rnn_LD_path).resolve()
    args.rnn_skip_path = Path(args.rnn_skip_path).resolve()
    args.rnn_skip_LD_path = Path(args.rnn_skip_LD_path).resolve()

    behavior_key, memoryless_key, memoryless_LD_key, memoryless_skip_key, memoryless_skip_LD_key, rnn_key, rnn_LD_key, rnn_skip_key, rnn_skip_LD_key, key = jax.random.split(key, 10)

    env, env_params, behavior_args, behavior_network, behavior_ts = load_train_state(behavior_key, args.behavior_path,
                                                                                     update_idx_to_take=args.update_idx_to_take,
                                                                                     best_over_rng=True)
    _, _, memoryless_args, memoryless_network, memoryless_ts = load_train_state(memoryless_key, args.memoryless_path,
                                                                                update_idx_to_take=args.update_idx_to_take,
                                                                                best_over_rng=True)
    _, _, memoryless_LD_args, memoryless_LD_network, memoryless_LD_ts = load_train_state(memoryless_LD_key, args.memoryless_LD_path,
                                                                                        update_idx_to_take=args.update_idx_to_take,
                                                                                        best_over_rng=True)
    _, _, memoryless_skip_args, memoryless_skip_network, memoryless_skip_ts = load_train_state(memoryless_skip_key, args.memoryless_skip_path,
                                                                                              update_idx_to_take=args.update_idx_to_take,
                                                                                              best_over_rng=True)
    _, _, memoryless_skip_LD_args, memoryless_skip_LD_network, memoryless_skip_LD_ts = load_train_state(memoryless_skip_LD_key, args.memoryless_skip_LD_path,
                                                                                                        update_idx_to_take=args.update_idx_to_take,
                                                                                                        best_over_rng=True)
    _, _, rnn_args, rnn_network, rnn_ts = load_train_state(rnn_key, args.rnn_path,
                                                              update_idx_to_take=args.update_idx_to_take,
                                                              best_over_rng=True)
    _, _, rnn_LD_args, rnn_LD_network, rnn_LD_ts = load_train_state(rnn_LD_key, args.rnn_LD_path,
                                                                    update_idx_to_take=args.update_idx_to_take,
                                                                    best_over_rng=True)
    _, _, rnn_skip_args, rnn_skip_network, rnn_skip_ts = load_train_state(rnn_skip_key, args.rnn_skip_path,
                                                                        update_idx_to_take=args.update_idx_to_take,
                                                                        best_over_rng=True)
    _, _, rnn_skip_LD_args, rnn_skip_LD_network, rnn_skip_LD_ts = load_train_state(rnn_skip_LD_key, args.rnn_skip_LD_path,
                                                                                  update_idx_to_take=args.update_idx_to_take,
                                                                                  best_over_rng=True)

    _env_step = partial(ppo_pocman_step, behavior_network=behavior_network,
                        memoryless_network=memoryless_network, memoryless_LD_network=memoryless_LD_network,
                        memoryless_skip_network=memoryless_skip_network, memoryless_skip_LD_network=memoryless_skip_LD_network,
                        rnn_network=rnn_network, rnn_LD_network=rnn_LD_network,
                        rnn_skip_network=rnn_skip_network, rnn_skip_LD_network=rnn_skip_LD_network,
                        env=env, env_params=env_params)
    _env_step = scan_tqdm(steps_to_collect)(_env_step)

    ckpts = {
        'behavior': {'args': behavior_args, 'ts': behavior_ts, 'path': args.behavior_path},
        'memoryless': {'args': memoryless_args, 'ts': memoryless_ts, 'path': args.memoryless_path},
        'memoryless_LD': {'args': memoryless_LD_args, 'ts': memoryless_LD_ts, 'path': args.memoryless_LD_path},
        'memoryless_skip': {'args': memoryless_skip_args, 'ts': memoryless_skip_ts, 'path': args.memoryless_skip_path},
        'memoryless_skip_LD': {'args': memoryless_skip_LD_args, 'ts': memoryless_skip_LD_ts, 'path': args.memoryless_skip_LD_path},
        'rnn': {'args': rnn_args, 'ts': rnn_ts, 'path': args.rnn_path},
        'rnn_LD': {'args': rnn_LD_args, 'ts': rnn_LD_ts, 'path': args.rnn_LD_path},
        'rnn_skip': {'args': rnn_skip_args, 'ts': rnn_skip_ts, 'path': args.rnn_skip_path},
        'rnn_skip_LD': {'args': rnn_skip_LD_args, 'ts': rnn_skip_LD_ts, 'path': args.rnn_skip_LD_path}
    }

    def collect(rng):
        # INIT ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)

        # init hidden state
        init_behavior_hstate = ScannedRNN.initialize_carry(args.num_envs, behavior_args['hidden_size'])
        init_rnn_hstate = ScannedRNN.initialize_carry(args.num_envs, rnn_args['hidden_size'])
        init_rnn_LD_hstate = ScannedRNN.initialize_carry(args.num_envs, rnn_LD_args['hidden_size'])
        init_runner_state = (
            behavior_ts,
            memoryless_ts,
            memoryless_LD_ts,
            memoryless_skip_ts,
            memoryless_skip_LD_ts,
            rnn_ts,
            rnn_LD_ts,
            rnn_skip_ts,
            rnn_skip_LD_ts,
            env_state,
            obsv,
            jnp.zeros(args.num_envs, dtype=bool),
            init_behavior_hstate,
            init_rnn_hstate,
            init_rnn_LD_hstate,
            _rng,
        )

        runner_state, dataset = jax.lax.scan(
            _env_step, init_runner_state, jnp.arange(steps_to_collect), steps_to_collect
        )

        # Now we flatten back down
        flat_dataset = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), dataset)

        return flat_dataset

    return collect, ckpts


if __name__ == "__main__":
    # jax.disable_jit(True)
    args = CollectHyperparams().parse_args()
    jax.config.update('jax_platform_name', args.platform)

    key = jax.random.PRNGKey(args.seed)
    make_key, collect_key, key = jax.random.split(key, 3)

    collect_fn, ckpt_info = make_collect(args, make_key)
    collect_fn = jax.jit(collect_fn)

    dataset = collect_fn(collect_key)

    def path_to_str(d: dict):
        for k, v in d.items():
            if isinstance(v, Path):
                d[k] = str(v)
            elif isinstance(v, dict):
                path_to_str(v)


    to_save = {
        'dataset': dataset,
        'args': args.as_dict(),
        'ckpt': ckpt_info,
    }
    path_to_str(to_save)

    save_path = args.behavior_path.parent / \
                f'dataset'

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(to_save)

    print(f"Saving results to {save_path}")
    orbax_checkpointer.save(save_path, to_save, save_args=save_args)

    print("Done.")