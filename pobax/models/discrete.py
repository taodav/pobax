import distrax
import flax.linen as nn
import jax.numpy as jnp
from jax._src.nn.initializers import orthogonal, constant
import numpy as np

from .network import SimpleNN, ScannedRNN, SmallImageCNN, RNNApproximator
from .value import Critic
from .jumanji import *


class DiscreteActor(nn.Module):
    action_dim: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)
        return pi


# class DiscreteActorCritic(nn.Module):
#     action_dim: int
#     fix_rnn: bool = False
#     horizon: int = 3
#     hidden_size: int = 128
#     depth: int = 3
#     double_critic: bool = False

#     @nn.compact
#     def __call__(self, _, x):
#         obs, dones = x
#         if self.fix_rnn:
#             _, embedding = FixedHorizonPlanningRNN(hidden_size=self.hidden_size, horizon=self.horizon)(obs)
#         else:
#             embedding = SimpleNN(hidden_size=self.hidden_size, depth=self.depth)(obs)

#         actor = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)
#         pi = actor(embedding)

#         critic = Critic(hidden_size=self.hidden_size)

#         if self.double_critic:
#             critic = nn.vmap(Critic,
#                              variable_axes={'params': 0},
#                              split_rngs={'params': True},
#                              in_axes=None,
#                              out_axes=2,
#                              axis_size=2)(hidden_size=self.hidden_size)

#         v = critic(embedding)

#         return _, pi, jnp.squeeze(v, axis=-1)
    
class DiscreteActorCritic(nn.Module):
    action_dim: int
    approximator: str = 'mlp'
    horizon: int = 3
    hidden_size: int = 128
    depth: int = 3
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        if self.approximator == 'rnn':
            embedding = nn.Dense(self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
            embedding = nn.relu(embedding)
            rnn_in = (embedding, dones)
            hidden, embedding = RNNApproximator(hidden_size=self.hidden_size, horizon=self.horizon)(hidden, rnn_in)
        else:
            embedding = SimpleNN(hidden_size=self.hidden_size, depth=self.depth)(obs)

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
        print(f'embedding shape: {embedding.shape}')
        v = critic(embedding)
        return hidden, pi, jnp.squeeze(v, axis=-1)


class DiscreteActorCriticRNN(nn.Module):
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
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

        print(f'embeeding shape: {embedding.shape}')
        v = critic(embedding)

        return hidden, pi, jnp.squeeze(v, axis=-1)


class ImageDiscreteActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, _, x):
        obs, dones = x

        embedding = SmallImageCNN(hidden_size=self.hidden_size)(obs)
        embedding = nn.relu(embedding)

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

        return _, pi, jnp.squeeze(v, axis=-1)


class ImageDiscreteActorCriticRNN(nn.Module):
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        embedding = SmallImageCNN(hidden_size=self.hidden_size)(obs)
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


class BattleShipActorCriticRNN(nn.Module):
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        # Obs is a t x b x obs_size array.
        obs, dones = x
        hit = obs[..., 0:1]
        valid_action_mask = obs[..., 1:self.action_dim + 1]
        obs = jnp.concatenate([hit, obs[..., self.action_dim + 1:]], axis=-1)

        embedding = nn.Dense(
            2 * self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        embedding = jnp.concatenate((hit, embedding), axis=-1)
        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)

        # MLP actor
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # Do masking here for invalid actions.
        actor_mean = actor_mean * valid_action_mask + (1 - valid_action_mask) * (-1e6)

        pi = distrax.Categorical(logits=actor_mean)

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


class BattleShipActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        # Obs is a t x b x obs_size array.
        obs, dones = x
        hit = obs[..., 0:1]
        valid_action_mask = obs[..., 1:self.action_dim + 1]
        obs = jnp.concatenate([hit, obs[..., self.action_dim + 1:]], axis=-1)

        embedding = nn.Dense(
            2 * self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        embedding = jnp.concatenate((hit, embedding), axis=-1)
        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(embedding)
        embedding = nn.relu(embedding)

        # MLP actor
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            embedding
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        # Do masking here for invalid actions.
        actor_mean = actor_mean * valid_action_mask + (1 - valid_action_mask) * (-1e6)

        pi = distrax.Categorical(logits=actor_mean)

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
    
jumanji_cnn_actor_dict = {
    "Game2048-v1": (Game2048CNN, Game2048Actor),
    "Sokoban-v0": (SokobanCNN, SokobanActor),
    "Snake-v1": (SnakeCNN, SnakeActor),
}

class JumanjiActorCriticRNN(nn.Module):
    env_name: str
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        cnn, Actor = jumanji_cnn_actor_dict[self.env_name]

        embedding = cnn(hidden_size=self.hidden_size)(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)

        actor = Actor(self.action_dim, hidden_size=self.hidden_size)
        pi = actor(embedding, obs)

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

class JumanjiActorCritic(nn.Module):
    env_name: str
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, _, x):
        obs, dones = x
        cnn, Actor = jumanji_cnn_actor_dict[self.env_name]
        embedding = cnn(hidden_size=self.hidden_size)(obs)
        embedding = nn.relu(embedding)

        actor = Actor(self.action_dim, hidden_size=self.hidden_size)
        pi = actor(embedding, obs)

        critic = Critic(hidden_size=self.hidden_size)

        if self.double_critic:
            critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)
        v = critic(embedding)
        return _, pi, jnp.squeeze(v, axis=-1)