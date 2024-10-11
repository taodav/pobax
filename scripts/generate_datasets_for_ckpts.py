from pathlib import Path

import chex
from flax.training import orbax_utils
import gymnax
from gymnax.wrappers.purerl import FlattenObservationWrapper, LogWrapper
import jax
import jax.numpy as jnp
import orbax.checkpoint
import flashbax as fbx

from pobax.algos.dqn import QNetwork, TimeStep


# epsilon-greedy exploration
def eps_greedy_exploration(rng, q_vals, eps: float):
    rng_a, rng_e = jax.random.split(
        rng, 2
    )  # a key for sampling random actions and one for picking
    greedy_actions = jnp.argmax(q_vals, axis=-1)  # get the greedy actions
    chosed_actions = jnp.where(
        jax.random.uniform(rng_e, greedy_actions.shape)
        < eps,  # pick the actions that should be random
        jax.random.randint(
            rng_a, shape=greedy_actions.shape, minval=0, maxval=q_vals.shape[-1]
        ),  # sample random actions,
        greedy_actions,
        )
    return chosed_actions


def make_collect(config: dict, n_samples: int, debug: bool = True):
    basic_env, env_params = gymnax.make(config["ENV_NAME"])
    env = FlattenObservationWrapper(basic_env)
    env = LogWrapper(env)

    network = QNetwork(action_dim=env.action_space(env_params).n, hidden_size=config["HIDDEN_SIZE"])

    # INIT BUFFER
    buffer = fbx.make_flat_buffer(
        max_length=n_samples,
        min_length=config["BUFFER_BATCH_SIZE"],
        sample_batch_size=config["BUFFER_BATCH_SIZE"],
        add_sequences=False,
    )
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )

    def collect(key: chex.PRNGKey, params: dict):
        dummy_rng = jax.random.PRNGKey(0)  # use a dummy rng here
        _action = basic_env.action_space().sample(dummy_rng)
        _, _env_state = env.reset(dummy_rng, env_params)
        _obs, _, _reward, _done, _ = env.step(dummy_rng, _env_state, _action, env_params)
        _timestep = TimeStep(obs=_obs, action=_action, reward=_reward, done=_done)
        init_buffer_state = buffer.init(_timestep)

        reset_key, key = jax.random.split(key)
        init_obs, env_state = env.reset(reset_key, env_params)

        def _env_step(runner_state, tstep):
            buffer_state, env_state, last_obs, rng = runner_state

            # STEP THE ENV
            rng, rng_a, rng_s = jax.random.split(rng, 3)
            q_vals = network.apply(params, last_obs)
            action = eps_greedy_exploration(
                rng_a, q_vals, config["EPSILON_FINISH"]
            )  # explore with epsilon greedy_exploration

            obs, env_state, reward, done, info = env.step(
                rng_s, env_state, action, env_params
            )

            # BUFFER UPDATE
            timestep = TimeStep(obs=last_obs, action=action, reward=reward, done=done)
            buffer_state = buffer.add(buffer_state, timestep)

            if debug:
                def callback(t):
                    if t % 100 == 0:
                        print(f"Collected timestep {t}")

                jax.debug.callback(callback, tstep)
            runner_state = (buffer_state, env_state, obs, rng)
            return runner_state, timestep

        steps_key, key = jax.random.split(key)
        runner_state = (init_buffer_state, env_state, init_obs, steps_key)

        final_runner_state, _ = jax.lax.scan(
            _env_step, runner_state, jnp.arange(n_samples), n_samples
        )
        final_buffer_state, final_env_state, final_obs, key = final_runner_state

        return final_buffer_state

    return collect


if __name__ == "__main__":
    results_path = Path('/Users/ruoyutao/Documents/pobax/results/Acrobot-v1_2024_37b0becb6279b2b7e878de87971959e9')
    n_samples = int(5e5)
    seed = 2024

    buffer_results_path = results_path.parent / (results_path.name + f"_buffer_{seed}")
    key = jax.random.PRNGKey(seed)

    # load our params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(results_path)

    config = restored['config']
    params = restored['train_states']['params']  # 1 x n_params x *param_size
    # remove the leading batch dimension
    params = jax.tree.map(lambda x: x[0], params)
    n_params = jax.tree.leaves(params)[0].shape[0]

    collect_fn = make_collect(config, n_samples, debug=True)

    vmap_collect = jax.jit(jax.vmap(collect_fn, in_axes=0))
    collect_keys = jax.random.split(key, n_params)

    buffer_state = vmap_collect(collect_keys, params)

    all_res = {
        'n_samples': n_samples,
        'seed': seed,
        'config': config,
        'params': params,
        'buffer_state': buffer_state
    }

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_res)
    print(f"Saving results to {buffer_results_path}")
    orbax_checkpointer.save(buffer_results_path, all_res, save_args=save_args)

    print("Done.")

