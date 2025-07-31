from gymnax.environments import environment, spaces
import navix as nx

from pobax.envs.jax.battleship import Battleship
from .actor_critic import ActorCritic

from .continuous import *
from .discrete import *

# def get_gymnax_network_fn(env: environment.Environment, env_params: environment.EnvParams,
#                           memoryless: bool = False):
#     if isinstance(env, Battleship) or ((hasattr(env, '_unwrapped') and isinstance(env._unwrapped, Battleship))):
#         network_fn = BattleShipActorCriticRNN
#         if memoryless:
#             network_fn = BattleShipActorCritic
#         action_size = env.action_space(env_params).n
#     elif isinstance(env.action_space(env_params), spaces.Discrete) or \
#             isinstance(env.action_space(env_params), nx.spaces.Discrete):
#         action_size = env.action_space(env_params).n
#
#         # Check whether we use image observations
#         obs_space_shape = env.observation_space(env_params).shape
#         if len(obs_space_shape) > 1:
#             # assert jnp.all(jnp.array(obs_space_shape[:-1]) == 5)
#             network_fn = ImageDiscreteActorCriticRNN
#             if memoryless:
#                 network_fn = ImageDiscreteActorCritic
#         else:
#             network_fn = DiscreteActorCriticRNN
#             if memoryless:
#                 network_fn = DiscreteActorCritic
#     elif isinstance(env.action_space(env_params), spaces.Box):
#         action_size = env.action_space(env_params).shape[0]
#         obs_space_shape = env.observation_space(env_params).shape
#         if len(obs_space_shape) > 1:
#             # assert jnp.all(jnp.array(obs_space_shape[:-1]) == 5)
#             network_fn = ImageContinuousActorCriticRNN
#             if memoryless:
#                 network_fn = ImageContinuousActorCritic
#         else:
#             network_fn = ContinuousActorCriticRNN
#             if memoryless:
#                 network_fn = ContinuousActorCritic
#     else:
#         raise NotImplementedError
#     return network_fn, action_size

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


class DiscreteActorCriticRNN(nn.Module):
    env: str
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False
    memoryless: bool = False
    is_discrete: bool = True
    is_image: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs.obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)

        actor = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)
        pi = actor(embedding)

        critic = Critic(hidden_size=self.hidden_size)

        if self.double_critic:
            critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)

        v = critic(embedding)

        return hidden, pi, jnp.squeeze(v, axis=-1)

def get_network_fn(env: environment.Environment, env_params: environment.EnvParams):
    network_fn = ActorCritic
    is_image = False
    observation_space = env.observation_space(env_params)
    if isinstance(env.action_space(env_params), spaces.Discrete):
        from pobax.models.discrete import DiscreteActor
        network_fn = DiscreteActorCriticRNN
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
