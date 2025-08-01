from gymnax.environments import environment, spaces

from pobax.envs.jax.battleship import Battleship
from .actor_critic import ActorCritic

from .continuous import *
from .discrete import *


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


# def get_network_fn(env: gym.Env,
#                    memoryless: bool = False):
#     obs_space = env.observation_space
#     action_space = env.action_space
#     obs_shape = obs_space.shape
#     if isinstance(action_space, gym.spaces.Discrete):
#         if len(obs_shape) > 1:
#             network_fn = ImageDiscreteActorCriticRNN
#             if memoryless:
#                 network_fn = ImageDiscreteActorCritic
#             action_size = action_space.n
#         else:
#             network_fn = DiscreteActorCriticRNN
#             if memoryless:
#                 network_fn = DiscreteActorCritic
#             action_size = action_space.n
#     else:
#         action_shape = action_space.shape
#         action_size = action_shape[0]
#
#         # Check whether we use image observations
#         if len(obs_shape) > 1:
#             # image observations
#             network_fn = ImageContinuousActorCriticRNN
#             if memoryless:
#                 network_fn = ImageContinuousActorCritic
#         else:
#             network_fn = ContinuousActorCriticRNN
#             if memoryless:
#                 network_fn = ContinuousActorCritic
#
#     return network_fn, obs_shape, action_size
