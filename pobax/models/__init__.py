from typing import Union

import gymnasium as gym
from gymnax.environments import environment, spaces
import navix as nx

from pobax.envs.jax.battleship import Battleship

from .continuous import *
from .discrete import *

def get_gymnax_network_fn(env: environment.Environment, env_params: environment.EnvParams,
                          memoryless: bool = False):
    if isinstance(env, Battleship) or ((hasattr(env, '_unwrapped') and isinstance(env._unwrapped, Battleship))):
        network_fn = BattleShipActorCriticRNN
        if memoryless:
            network_fn = BattleShipActorCritic
        action_size = env.action_space(env_params).n
    elif isinstance(env.action_space(env_params), spaces.Discrete) or \
            isinstance(env.action_space(env_params), nx.spaces.Discrete):
        action_size = env.action_space(env_params).n

        # Check whether we use image observations
        obs_space_shape = env.observation_space(env_params).shape
        if len(obs_space_shape) > 1:
            # assert jnp.all(jnp.array(obs_space_shape[:-1]) == 5)
            network_fn = ImageDiscreteActorCriticRNN
            if memoryless:
                network_fn = ImageDiscreteActorCritic
        else:
            network_fn = DiscreteActorCriticRNN
            if memoryless:
                network_fn = DiscreteActorCritic
    elif isinstance(env.action_space(env_params), spaces.Box):
        action_size = env.action_space(env_params).shape[0]
        obs_space_shape = env.observation_space(env_params).shape
        if len(obs_space_shape) > 1:
            # assert jnp.all(jnp.array(obs_space_shape[:-1]) == 5)
            network_fn = ImageContinuousActorCriticRNN
            if memoryless:
                network_fn = ImageContinuousActorCritic
        else:
            network_fn = ContinuousActorCriticRNN
            if memoryless:
                network_fn = ContinuousActorCritic
    else:
        raise NotImplementedError
    return network_fn, action_size

def get_transformer_network_fn(env: environment.Environment, env_params: environment.EnvParams):
    if isinstance(env, Battleship) or ((hasattr(env, '_unwrapped') and isinstance(env._unwrapped, Battleship))):
        network_fn = BattleShipActorCriticTransformer
        action_size = env.action_space(env_params).n
    elif isinstance(env.action_space(env_params), spaces.Discrete):
        action_size = env.action_space(env_params).n

        # Check whether we use image observations
        obs_space_shape = env.observation_space(env_params).shape
        if len(obs_space_shape) > 1:
            # assert jnp.all(jnp.array(obs_space_shape[:-1]) == 5)
            network_fn = ImageDiscreteActorCriticTransformer
        else:
            network_fn = DiscreteActorCriticTransformer
    elif isinstance(env.action_space(env_params), spaces.Box):
        action_size = env.action_space(env_params).shape[0]
        obs_space_shape = env.observation_space(env_params).shape
        if len(obs_space_shape) > 1:
            # assert jnp.all(jnp.array(obs_space_shape[:-1]) == 5)
            network_fn = ImageContinuousActorCriticTransformer
        else:
            network_fn = ContinuousActorCriticTransformer
    else:
        raise NotImplementedError
    return network_fn, action_size

def get_network_fn(env: environment.Environment, env_params: environment.EnvParams):
    network_fn = ActorCritic
    is_image = False
    observation_space = env.observation_space(env_params)
    if isinstance(env.action_space(env_params), spaces.Discrete):
        action_size = env.action_space(env_params).n
        is_discrete = True
    elif isinstance(env.action_space(env_params), spaces.Box):
        action_size = env.action_space(env_params).shape[0]
        is_discrete = False
    for subspace in observation_space.spaces.values():
        if len(subspace.shape) > 1:
            is_image = True
            break
    return network_fn, action_size, is_image, is_discrete
    