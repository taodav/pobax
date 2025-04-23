from functools import partial
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import optax
import orbax.checkpoint

from pobax.algos.gd_ppo import GDPPO, GDTrainState, env_step
from pobax.config import GDPPOHyperparams
from pobax.envs import get_env
from pobax.models import ScannedRNN
from pobax.models.actor_critic import CumulantGammaNetwork, ActorCritic


def load_train_state(fpath: Path, key: chex.PRNGKey):
    env_key, key = jax.random.split(key)
    # load our params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(fpath)
    args = restored['args']
    args = GDPPOHyperparams().from_dict(args)

    env, env_params = get_env(args.env, env_key,
                              gamma=args.gamma,
                              normalize_image=False,
                              perfect_memory=args.perfect_memory,
                              action_concat=args.action_concat,
                              reward_concat=args.reward_concat)

    cumulant_size = None
    if args.cumulant_type == 'hs':
        cumulant_size = args.cumulant_map_size
    elif args.cumulant_type == 'rew':
        cumulant_size = 1
    elif args.cumulant_type == 'hs_rew':
        cumulant_size = args.cumulant_map_size + 1

    network = ActorCritic(env.action_space(env_params),
                          memoryless=args.memoryless,
                          double_critic=args.double_critic,
                          hidden_size=args.hidden_size,
                          cumulant_size=cumulant_size)

    cumulant_gamma_network = CumulantGammaNetwork(cumulant_size=cumulant_size,
                                                  gamma_type=args.gamma_type)

    agent = GDPPO(network, cumulant_gamma_network,
                  double_critic=args.double_critic,
                  ld_weight=args.ld_weight.item(),
                  alpha=args.alpha.item(),
                  vf_coeff=args.vf_coeff.item(),
                  clip_eps=args.clip_eps,
                  entropy_coeff=args.entropy_coeff,
                  ld_exploration_bonus_scale=args.ld_exploration_bonus_scale)

    ts_dict = jax.tree.map(lambda x: x[0, 0, 0, 0, 0, 0, 0], restored['final_train_state'])
    tx = optax.adam(args.lr)

    train_state = GDTrainState.create(
        apply_fn=agent.network.apply,
        params=ts_dict['params'],
        tx=tx,
        cumulant_gamma_params=ts_dict['cumulant_gamma_params']
    )

    return env, env_params, args, agent, train_state


if __name__ == "__main__":
    jax.disable_jit(True)

    rng = jax.random.key(2024)
    n_envs = 2
    n_steps = int(1e3)
    rng, load_key = jax.random.split(rng)

    ckpt_path = Path('/Users/ruoyutao/Documents/pobax/results/test_gd_ppo/tmaze_5_seed(2024)_time(20250423-150113)_e3dba12c8888a375fe47a98f570c4b43')

    env, env_params, args, agent, ts = load_train_state(ckpt_path, load_key)

    # initialize functions
    _env_step = partial(env_step, agent=agent, env=env, env_params=env_params)


    # INIT NETWORK
    rng, _rng = jax.random.split(rng)
    init_x = (
        jnp.zeros(
            (1, args.num_envs, *env.observation_space(env_params).shape)
        ),
        jnp.zeros((1, args.num_envs)),
    )
    init_hstate = ScannedRNN.initialize_carry(args.num_envs, args.hidden_size)

    rng, _rng = jax.random.split(rng)
    reset_rng = jax.random.split(_rng, args.num_envs)
    init_obsv, env_state = env.reset(reset_rng, env_params)

    init_runner_state = (
        ts,
        env_state,
        init_obsv,
        jnp.zeros(args.num_envs, dtype=bool),
        init_hstate,
        _rng,
    )
    runner_state, traj_batch = jax.lax.scan(
        _env_step, init_runner_state, None, n_steps
    )
    print()



