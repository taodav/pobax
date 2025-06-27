from typing import NamedTuple

from collections import deque
from dataclasses import replace
from functools import partial
import inspect
from time import time
# import os
# os.environ["CRAFTAX_RELOAD_TEXTURES"] = "True"

import chex
import flax.training.train_state
from flax.training.train_state import TrainState
from flax.training import orbax_utils
import jax
import jax.numpy as jnp
import numpy as np
import optax
import orbax.checkpoint

from pobax.config import TransformerHyperparams
from pobax.envs import get_env
from pobax.envs.wrappers.gymnax import LogEnvState
from pobax.models import get_transformer_network_fn
from pobax.utils.file_system import get_results_path

class Transition(NamedTuple):
    done: jnp.ndarray
    action: jnp.ndarray
    value: jnp.ndarray
    reward: jnp.ndarray
    log_prob: jnp.ndarray
    memories_mask:jnp.ndarray
    memories_indices:jnp.ndarray
    obs: jnp.ndarray
    info: jnp.ndarray

class PPO:
    def __init__(self, network,
                 double_critic: bool = False,
                 ld_weight: float = 0.,
                 alpha: float = 0.,
                 vf_coeff: float = 0.,
                 entropy_coeff: float = 0.01,
                 clip_eps: float = 0.2):
        self.network = network
        self.double_critic = double_critic
        self.ld_weight = ld_weight
        self.alpha = alpha
        self.vf_coeff = vf_coeff
        self.entropy_coeff = entropy_coeff
        self.clip_eps = clip_eps
        self.loss = jax.jit(self.loss)

    def loss(self, params, traj_batch,memories_batch, gae, targets):
                        
        # USE THE CACHED MEMORIES ONLY FROM THE FIRST STEP OF A WINDOW GRAD Because all other will be computed again here.
        #construct the memory batch from memory indices
        memories_batch=batch_indices_select(memories_batch,traj_batch.memories_indices[:,::args.window_grad]) 
        memories_batch=batchify(memories_batch)
                        
                        
        #CREATE THE MASK FOR WINDOW GRAD (have to take the one from the batch and roll them to match the steps it attends
        memories_mask=traj_batch.memories_mask.reshape((-1,args.window_grad,)+traj_batch.memories_mask.shape[2:])
        memories_mask=jnp.swapaxes(memories_mask,1,2)
        #concatenate with 0s to fill before the roll
        memories_mask=jnp.concatenate((memories_mask,jnp.zeros(memories_mask.shape[:-1]+(args.window_grad-1,),dtype=jnp.bool_)),axis=-1)
        #roll of different value for each step to match the right
        memories_mask=roll_vmap(memories_mask,jnp.arange(0,args.window_grad),-1)

        #RESHAPE
        obs=traj_batch.obs
        obs=obs.reshape((-1, args.window_grad,)+obs.shape[2:])

        traj_batch,targets,gae=jax.tree_util.tree_map(lambda x : jnp.reshape(x,(-1,args.window_grad)+x.shape[2:]),(traj_batch,targets,gae))    
                        
        # NETWORK OUTPUT
        pi, value = self.network.apply(params,memories_batch, obs,memories_mask,method=self.network.model_forward_train)
                        
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE VALUE LOSS
        value_pred_clipped = traj_batch.value + (
            value - traj_batch.value
        ).clip(-args.clip_eps, args.clip_eps)
        value_losses = jnp.square(value - targets)
        value_losses_clipped = jnp.square(value_pred_clipped - targets)
        value_loss = (
            0.5 * jnp.maximum(value_losses, value_losses_clipped).mean()
        )
        if self.double_critic:
            value_loss = self.ld_weight * (jnp.square(value[..., 0] - value[..., 1])).mean() + \
                         (1 - self.ld_weight) * value_loss

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)
        if self.double_critic:
            gae = (self.alpha * gae[..., 0] +
                   (1 - self.alpha) * gae[..., 1])
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
            jnp.clip(
                ratio,
                1.0 - args.clip_eps,
                1.0 + args.clip_eps,
            )
            * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
            loss_actor
            + self.vf_coeff * value_loss
            - self.entropy_coeff * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)

def env_step(runner_state, unused, env, network, env_params, args: TransformerHyperparams):
    train_state, env_state,memories,memories_mask,memories_mask_idx, last_obs,done,step_env_currentloop, rng = runner_state
                
    # reset memories mask and mask idx in cask of done otherwise mask will consider one more stepif not filled (if filled= 
    memories_mask_idx=jnp.where(done,args.window_mem ,jnp.clip(memories_mask_idx-1,0,args.window_mem))
    memories_mask=jnp.where(done[:,None,None,None],jnp.zeros((args.num_envs,args.num_heads,1,args.window_mem+1),dtype=jnp.bool_),memories_mask)
                
    #Update memories mask with the potential additional step taken into account at this step
    memories_mask_idx_ohot=jax.nn.one_hot(memories_mask_idx,args.window_mem+1)
    memories_mask_idx_ohot=memories_mask_idx_ohot[:,None,None,:].repeat(args.num_heads,1)
    memories_mask=jnp.logical_or(memories_mask, memories_mask_idx_ohot)

    # SELECT ACTION
    rng, _rng = jax.random.split(rng)
    pi, value,memories_out = network.apply(train_state.params , memories, last_obs,memories_mask,method=network.model_forward_eval)
    action = pi.sample(seed=_rng)
    log_prob = pi.log_prob(action)
                
    # ADD THE CACHED ACTIVATIONS IN MEMORIES FOR NEXT STEP
    memories=jnp.roll(memories,-1,axis=1).at[:,-1].set(memories_out)
                
    # STEP ENV
    rng, _rng = jax.random.split(rng)
    rng_step = jax.random.split(_rng, args.num_envs)
    obsv, env_state, reward, done, info =env.step(
        rng_step, env_state, action, env_params
    )
                             
    #COMPUTE THE INDICES OF THE FINAL MEMORIES THAT ARE TAKEN INTO ACCOUNT IN THIS STEP 
    # not forgeeting that we will concatenate the previous WINDOW_MEM to the NUM_STEPS so that even the first step will use some cached memory.
    #previous without this is attend to 0 which are masked but with reset happening if we start the num_steps loop during good to keep memory from previous
    memory_indices=jnp.arange(0,args.window_mem)[None,:]+step_env_currentloop*jnp.ones((args.num_envs,1),dtype=jnp.int32)
                
    transition = Transition(
        done, action, value, reward, log_prob, memories_mask.squeeze(),memory_indices, last_obs, info
    )
    runner_state = (train_state, env_state,memories, memories_mask, memories_mask_idx, obsv, done, step_env_currentloop+1, rng)
    return runner_state, (transition,memories_out)
            

def calculate_gae(traj_batch, last_val, last_done, gae_lambda, gamma):
    def _get_advantages(carry, transition):
        gae, next_value, next_done, gae_lambda = carry
        done, value, reward = transition.done, transition.value, transition.reward
        delta = reward + gamma * next_value * (1 - next_done) - value
        gae = delta + gamma * gae_lambda * (1 - next_done) * gae
        return (gae, value, done, gae_lambda), gae

    _, advantages = jax.lax.scan(_get_advantages,
                                 (jnp.zeros_like(last_val), last_val, last_done, gae_lambda),
                                 traj_batch, reverse=True, unroll=16)
    target = advantages + traj_batch.value
    return advantages, target

def filter_period_first_dim(x, n: int):
    if isinstance(x, jnp.ndarray) or isinstance(x, np.ndarray):
        return x[::n]

indices_select=lambda x,y:x[y]
batch_indices_select=jax.vmap(indices_select)
roll_vmap=jax.vmap(jnp.roll,in_axes=(-2,0,None),out_axes=-2)
batchify=lambda x: jnp.reshape(x,(x.shape[0]*x.shape[1],)+x.shape[2:])

def make_train(args: TransformerHyperparams, rand_key: jax.random.PRNGKey):
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
                                     action_concat=args.action_concat)
    if hasattr(env, 'gamma'):
        args.gamma = env.gamma

    assert hasattr(env_params, 'max_steps_in_episode')

    double_critic = args.double_critic

    network_fn, action_size = get_transformer_network_fn(env, env_params)
    network = network_fn(action_dim=action_size,
                         encoder_size=args.embed_size,
                         num_heads=args.num_heads,
                         qkv_features=args.qkv_features,
                         num_layers=args.num_layers,
                         double_critic=double_critic,
                         hidden_size=args.hidden_size,
                         gating=args.gating,
                         gating_bias=args.gating_bias)
    
    steps_filter = partial(filter_period_first_dim, n=args.steps_log_freq)
    update_filter = partial(filter_period_first_dim, n=args.update_log_freq)

    transition_axes_map = Transition(
        None, None, 2, None, None, None, None, None, None
    )
    _calculate_gae = calculate_gae
    if double_critic:
        # last_val is index 1 here b/c we squeezed earlier.
        _calculate_gae = jax.vmap(calculate_gae,
                                 in_axes=[transition_axes_map, 1, None, 0, None],
                                 out_axes=2)
    
    def train(vf_coeff, ld_weight, alpha, lambda1, lambda0, lr, rng):
        agent = PPO(network, double_critic=double_critic, ld_weight=ld_weight, alpha=alpha, vf_coeff=vf_coeff,
                    clip_eps=args.clip_eps, entropy_coeff=args.entropy_coeff)
        # initialize functions
        _env_step = partial(env_step, network=network, env=env, env_params=env_params, args=args)

        gae_lambda = jnp.array(lambda0)
        if double_critic:
            gae_lambda = jnp.array([lambda0, lambda1])

        def linear_schedule(count):
            frac = (
                    1.0
                    - (count // (args.num_minibatches * args.update_epochs))
                    / num_updates
            )
            return lr * frac
        
        # INIT NETWORK
        rng, _rng = jax.random.split(rng)
        init_obs = jnp.zeros((2,*env.observation_space(env_params).shape))
        init_memory=jnp.zeros((2,args.window_mem,args.num_layers,args.embed_size))
        init_mask=jnp.zeros((2,args.num_heads,1,args.window_mem+1),dtype=jnp.bool_)
        network_params = network.init(_rng, init_memory,init_obs,init_mask)

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

        # Reset ENV
        rng, _rng = jax.random.split(rng)
        reset_rng = jax.random.split(_rng, args.num_envs)
        obsv, env_state = env.reset(reset_rng, env_params)
        
        # TRAIN LOOP
        def _update_step(runner_state, i):
    
            #also copy the first memories in memories_previous before the new rollout to concatenate previous memories with new steps so that first steps of new have memories
            memories_previous=runner_state[2]
             
            #SCAN THE STEP TO GET THE TRANSITIONS AND CACHED MEMORIES
            runner_state, (traj_batch,memories_batch) = jax.lax.scan(
                _env_step, runner_state, jnp.arange(args.num_steps), args.num_steps
            )

            # CALCULATE ADVANTAGE
            train_state, env_state,memories,memories_mask,memories_mask_idx, last_obs,done,_, rng = runner_state
            _, last_val,_ = network.apply(train_state.params, memories,last_obs,memories_mask,method=network.model_forward_eval)

            advantages, targets = _calculate_gae(traj_batch, last_val, done, gae_lambda, args.gamma)
            
            # UPDATE NETWORK
            def _update_epoch(update_state, unused):
                def _update_minbatch(train_state, batch_info):
                
                    traj_batch, memories_batch, advantages, targets = batch_info

                    grad_fn = jax.value_and_grad(agent.loss, has_aux=True)
                    total_loss, grads = grad_fn(
                        train_state.params, traj_batch,memories_batch, advantages, targets
                    )
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, total_loss

                train_state, traj_batch,memories_batch, advantages, targets, rng = update_state
                rng, _rng = jax.random.split(rng)            
                #batch_size = config["MINIBATCH_SIZE"] * config["NUM_MINIBATCHES"]
                assert (
                     args.num_steps % args.window_grad==0
                ), "NUM_STEPS should be divi by WINDOW_GRAD to properly batch the window_grad"
                
                
                # PERMUTE ALONG THE NUM_ENVS ONLY NOT TO LOOSE TRACK FROM TEMPORAL 
                permutation = jax.random.permutation(_rng, args.num_envs)
                batch = (traj_batch,memories_batch, advantages, targets)
                batch = jax.tree_util.tree_map(
                    lambda x:  jnp.swapaxes(x,0,1),
                    batch,
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, permutation, axis=0), batch
                )
                
                #either create memory batch here but might be big  or send all the memeory to loss and do the things with the index in the loss
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, [args.num_minibatches, -1] + list(x.shape[1:])
                    ),
                    shuffled_batch,
                )
            
                train_state, total_loss = jax.lax.scan(
                    _update_minbatch, train_state, minibatches
                )

                update_state = (train_state, traj_batch,memories_batch, advantages, targets, rng)
                return update_state, total_loss
            
            
            #ADD PREVIOUS WINDOW_MEM To the current NUM_STEPS SO THAT FIRST STEPS USE MEMORIES FROM PREVIOUS
            # might be a better place to add the previous memory to the traj batch to make it faster ??? 
            #or another solution is to not add it but in training means that the first element might not look at info
            memories_batch=jnp.concatenate([jnp.swapaxes(memories_previous,0,1),memories_batch],axis=0)
            
            update_state = (
                train_state, 
                traj_batch,
                memories_batch, 
                advantages, 
                targets, 
                rng)
            
            update_state, loss_info = jax.lax.scan(
                _update_epoch, update_state, None, args.update_epochs
            )
            metric = traj_batch.info
            metric = jax.tree.map(steps_filter, metric)

            train_state = update_state[0]
            rng = update_state[-1]
            if args.debug:

                def callback(info):
                    timesteps = (
                            info["timestep"][info["returned_episode"]] * args.num_envs
                    )
                    avg_return_values = jnp.mean(info["returned_episode_returns"][info["returned_episode"]])
                    if len(timesteps) > 0:
                        print(
                            f"timesteps={timesteps[0]} - {timesteps[-1]}, avg episodic return={avg_return_values:.2f}"
                        )

                jax.debug.callback(callback, metric)
            
            runner_state = (train_state, env_state,memories,memories_mask,memories_mask_idx, last_obs,done,0, rng)
            
            return runner_state, metric

        # INITIALIZE the memories and memories mask 
        rng, _rng = jax.random.split(rng)
        memories=jnp.zeros((args.num_envs,args.window_mem,args.num_layers,args.embed_size))
        memories_mask=jnp.zeros((args.num_envs,args.num_heads,1,args.window_mem+1),dtype=jnp.bool_)
        #memories +1 bc will remove one 
        memories_mask_idx= jnp.zeros((args.num_envs,),dtype=jnp.int32)+(args.window_mem+1)
        done=jnp.zeros((args.num_envs,),dtype=jnp.bool_)
        
        runner_state = (train_state, env_state,memories,memories_mask,memories_mask_idx, obsv,done,0, _rng)
        runner_state, metric = jax.lax.scan(
            _update_step, runner_state, jnp.arange(num_updates), num_updates
        )
        # save metrics only every update_log_freq
        metric = jax.tree.map(update_filter, metric)
        res = {"runner_state": runner_state, "metric": metric}
        return res
    
    return train


if __name__ == "__main__":
    # jax.disable_jit(True)
    args = TransformerHyperparams().parse_args()
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

    # Save all results with Orbax
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    save_args = orbax_utils.save_args_from_target(all_results)

    print(f"Saving results to {results_path}")
    orbax_checkpointer.save(results_path, all_results, save_args=save_args)
    print("Done.")
