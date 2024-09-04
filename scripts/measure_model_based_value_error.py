from functools import partial
import os
from pathlib import Path

from flax.training.train_state import TrainState
import gymnasium as gym
import hydra
from hydra import compose, initialize
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint as ocp
import tensorboard

from pobax.agents.tdmpc2 import TDMPC2
from pobax.envs import make_env
from pobax.models.network import TDMPC2ImageCNN
from pobax.models.world_model import WorldModel, vec_encoder
from pobax.utils.buffer import SequentialReplayBuffer

if __name__ == "__main__":
    results_path = Path('../results/cheetah-run_front_s0/2024-09-03_13-24-30')
    horizon = 100
    batch_size = 32


    checkpoint_path = results_path / 'checkpoint'
    hydra_path = results_path / '.hydra'

    with initialize(version_base=None, config_path=str(hydra_path)):
        cfg = compose(config_name='config')

    env_config = cfg['env']
    encoder_config = cfg['encoder']
    model_config = cfg['world_model']
    tdmpc_config = cfg['tdmpc2']

    ##############################
    # Environment setup
    ##############################
    vector_env_cls = gym.vector.AsyncVectorEnv if env_config.asynchronous else gym.vector.SyncVectorEnv
    env = vector_env_cls(
        [
            partial(make_env, env_config, seed)
            for seed in range(cfg.seed, cfg.seed+env_config.num_envs)
        ])
    np.random.seed(cfg.seed)
    rng = jax.random.PRNGKey(cfg.seed)

    ##############################
    # Agent setup
    ##############################
    dtype = jnp.dtype(model_config.dtype)
    rng, model_key, encoder_key = jax.random.split(rng, 3)

    encoder_module = vec_encoder(encoder_config, model_config, dtype)
    if 'dmc' in env_config and 'rgb' in env_config.dmc['obs_type']:
        encoder_module = TDMPC2ImageCNN(simnorm_dim=model_config.simnorm_dim,
                                        num_channels=encoder_config.num_channels)

    if encoder_config.tabulate:
        print("Encoder")
        print("--------------")
        print(encoder_module.tabulate(jax.random.key(0),
                                      env.observation_space.sample(), compute_flops=True))

    ##############################
    # Replay buffer setup
    ##############################
    dummy_obs, _ = env.reset()
    # dummy_obs = env.reset()
    dummy_action = env.action_space.sample()
    dummy_next_obs, dummy_reward, dummy_term, dummy_trunc, _ = \
        env.step(dummy_action)
    replay_buffer = SequentialReplayBuffer(
        capacity=cfg.max_steps//env_config.num_envs,
        num_envs=env_config.num_envs,
        seed=cfg.seed,
        dummy_input=dict(
            observation=dummy_obs,
            action=dummy_action,
            reward=dummy_reward,
            next_observation=dummy_next_obs,
            terminated=dummy_term,
            truncated=dummy_trunc)
    )

    encoder = TrainState.create(
        apply_fn=encoder_module.apply,
        params=encoder_module.init(encoder_key, dummy_obs)['params'],
        tx=optax.chain(
            optax.zero_nans(),
            optax.clip_by_global_norm(model_config.max_grad_norm),
            optax.adam(encoder_config.learning_rate),
        ))

    model = WorldModel.create(
        action_dim=np.prod(env.get_wrapper_attr('single_action_space').shape),
        encoder=encoder,
        **model_config,
        key=model_key)
    if model.action_dim >= 20:
        tdmpc_config.mppi_iterations += 2

    agent = TDMPC2.create(world_model=model, **tdmpc_config)

    with ocp.CheckpointManager(
            checkpoint_path.absolute(), item_names=(
                    'agent', 'global_step', 'buffer_state')) as mngr:
        print('Checkpoint folder found, restoring from', mngr.latest_step())
        abstract_buffer_state = jax.tree.map(
            ocp.utils.to_shape_dtype_struct, replay_buffer.get_state()
        )
        restored = mngr.restore(mngr.latest_step(),
                                args=ocp.args.Composite(
                                    agent=ocp.args.StandardRestore(agent),
                                    global_step=ocp.args.JsonRestore(),
                                    buffer_state=ocp.args.StandardRestore(abstract_buffer_state),
                                )
                                )
        agent, global_step = restored.agent, restored.global_step
        replay_buffer.restore(restored.buffer_state)

    # TODO: we need to collect samples from the start of an episode
    # TODO: OR start with a single z, then go from there.
    # sample = replay_buffer.sample(batch_size, horizon)
    # we do rollouts here
    # _z = sample
    # for t in range(self.horizon - 1):
    #     policy_actions = policy_actions.at[t].set(
    #         self.model.sample_actions(_z, self.model.policy_model.params, key=prior_keys[t])[0])
    #     _z = self.model.next(
    #         _z, policy_actions[t], self.model.dynamics_model.params)

    prev_plan = (
        jnp.zeros((env_config.num_envs, agent.horizon, agent.model.action_dim)),
        jnp.full((env_config.num_envs, agent.horizon,
                  agent.model.action_dim), agent.max_plan_std)
    )
    observation, _ = env.reset(seed=cfg.seed)
    for i in range(horizon):
        rng, action_key = jax.random.split(rng)
        prev_plan = (prev_plan[0],
                     jnp.full_like(prev_plan[1], agent.max_plan_std))
        action, prev_plan = agent.act(
            observation, prev_plan=prev_plan, train=True, key=action_key)

        next_observation, reward, terminated, truncated, info = env.step(action)

    print()

