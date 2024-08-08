
from gymnax.environments import environment, spaces
from pobax.envs.battleship import Battleship

from .continuous import *
from .discrete import *
from pobax.envs import jumanji_envs


def get_network_fn(env: environment.Environment, env_params: environment.EnvParams,
                   memoryless: bool = False):
    if isinstance(env, Battleship) or (hasattr(env, '_unwrapped') and isinstance(env._unwrapped, Battleship)):
        network_fn = BattleShipActorCriticRNN
        action_size = env.action_space(env_params).n
        if memoryless:
            network_fn = BattleShipActorCritic
    elif env.env_name in jumanji_envs:
        action_size = env.action_space(env_params).n
        network_fn = JumanjiActorCriticRNN
        if memoryless:
            network_fn = JumanjiActorCritic
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
