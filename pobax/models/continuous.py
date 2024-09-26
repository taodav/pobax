import distrax
import flax.linen as nn
import jax.numpy as jnp
from jax._src.nn.initializers import orthogonal, constant
import numpy as np

from .network import SimpleNN, ScannedRNN, SmallImageCNN, FullImageCNN
from .value import Critic


class ContinuousActor(nn.Module):
    action_dim: int
    hidden_size: int = 128
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x):
        if self.activation == "relu":
            activation = nn.relu
        else:
            activation = nn.tanh

        actor_mean = nn.Dense(
            2 * self.hidden_size, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)
        )(x)
        actor_mean = activation(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        actor_logtstd = self.param("log_std", nn.initializers.zeros, (self.action_dim,))
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        return pi


class ContinuousActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 128
    activation: str = "tanh"
    double_critic: bool = False

    @nn.compact
    def __call__(self, _, x):
        obs, dones = x

        embedding = SimpleNN(hidden_size=self.hidden_size)(obs)

        actor = ContinuousActor(self.action_dim, hidden_size=self.hidden_size,
                                activation=self.activation)
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

class ImageContinuousActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 128
    activation: str = "tanh"
    double_critic: bool = False

    @nn.compact
    def __call__(self, _, x):
        obs, dones = x

        embedding = SmallImageCNN(hidden_size=self.hidden_size)(obs)
        embedding = nn.relu(embedding)

        actor = ContinuousActor(self.action_dim, hidden_size=self.hidden_size,
                                activation=self.activation)
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

class ContinuousActorCriticRNN(nn.Module):
    action_dim: int
    hidden_size: int = 128
    activation: str = "tanh"
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

        actor = ContinuousActor(self.action_dim, hidden_size=self.hidden_size,
                                activation=self.activation)
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


class ImageContinuousActorCriticRNN(nn.Module):
    """
    Image Continuous Actor Critic RNN uses a different class of CNNs!
    (for larger RGB images)
    """
    action_dim: int
    hidden_size: int = 128
    activation: str = "tanh"
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = FullImageCNN(hidden_size=self.hidden_size)(obs)
        embedding = nn.relu(embedding)

        rnn_in = (embedding, dones)
        hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)

        actor = ContinuousActor(self.action_dim, hidden_size=self.hidden_size,
                                activation=self.activation)
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
