
import gymnasium as gym
from gymnax.environments import environment, spaces
from sympy.physics.units import action

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
    elif isinstance(env.action_space(env_params), spaces.Discrete):
        action_size = env.action_space(env_params).n

        # Check whether we use image observations
        obs_space_shape = env.observation_space(env_params).shape
        if len(obs_space_shape) > 1:
            assert jnp.all(jnp.array(obs_space_shape[:-1]) == 5)
            network_fn = ImageDiscreteActorCriticRNN
            if memoryless:
                network_fn = ImageDiscreteActorCritic

        else:
            network_fn = DiscreteActorCriticRNN
            if memoryless:
                network_fn = DiscreteActorCritic
    elif isinstance(env.action_space(env_params), spaces.Box):
        action_size = env.action_space(env_params).shape[0]
        network_fn = ContinuousActorCriticRNN
        if memoryless:
            network_fn = ContinuousActorCritic
    else:
        raise NotImplementedError
    return network_fn, action_size


def get_network_fn(env: gym.Env, memoryless: bool = False):
    # TODO
    if isinstance(env.action_space, gym.spaces.Box):
        action_shape = env.action_space.shape
        if len(action_shape) > 1:
            # vec env
            action_size = action_shape[-1]

        # Check whether we use image observations
        if isinstance(env.observation_space, gym.spaces.Dict):
            network_fn = ImageContinuousActorCriticRNN
            if memoryless:
                network_fn = ImageContinuousActorCritic
    else:
        raise NotImplementedError

    return network_fn, action_size

