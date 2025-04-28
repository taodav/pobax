from typing import Union
from flax import linen as nn
from jax import numpy as jnp
from jax._src.nn.initializers import orthogonal, constant
from gymnax.environments import spaces

from pobax.models.actor import Actor
from pobax.models.discrete import DiscreteActor
from pobax.models.continuous import ContinuousActor
from pobax.models.critic import Critic, GVF
from pobax.models.network import SimpleNN, ScannedRNN, FullImageCNN
from pobax.models.embedding import CNN


class ActorCritic(nn.Module):
    env_name: str
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False
    memoryless: bool = False
    is_discrete: bool = True
    is_image: bool = False

    def setup(self):
        if self.is_image:
            self.embedding = CNN(hidden_size=self.hidden_size)
        # elif 'battleship' in self.env_name:
        #     self.embedding = BattleshipEmbedding(hidden_size=self.hidden_size, action_dim=self.action_dim)
        elif not self.memoryless:
            self.embedding = nn.Sequential([
                nn.Dense(self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)),
                nn.relu
            ])
        else:
            self.embedding = SimpleNN(hidden_size=self.hidden_size)

        if not self.memoryless:
            self.memory = ScannedRNN(hidden_size=self.hidden_size)
        if self.is_discrete:
            self.actor = DiscreteActor(self.action_dim, hidden_size=self.hidden_size)
        else:
            self.actor = ContinuousActor(self.action_dim, hidden_size=self.hidden_size)

        if self.double_critic:
            self.critic = nn.vmap(Critic,
                                  variable_axes={'params': 0},
                                  split_rngs={'params': True},
                                  in_axes=None,
                                  out_axes=2,
                                  axis_size=2)(hidden_size=self.hidden_size)
        else:
            self.critic = Critic(hidden_size=self.hidden_size)

    def __call__(self, hidden, x):
        obs_dict, dones = x
        obs = obs_dict.obs
        action_mask = obs_dict.action_mask
        embedding = self.embedding(obs)
        if not self.memoryless:
            rnn_in = (embedding, dones)
            hidden, embedding = self.memory(hidden, rnn_in)

        pi = self.actor(embedding, action_mask=action_mask)
        v = self.critic(embedding)

        return hidden, pi, jnp.squeeze(v, axis=-1)


class GVFActorCritic(nn.Module):
    action_space: Union[spaces.Discrete, spaces.Box]
    hidden_size: int = 128
    cumulant_size: int = None
    memoryless: bool = False
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        if len(obs.shape) > 3:
            obs_encoding = FullImageCNN(hidden_size=self.hidden_size)(obs)
            obs_encoding = nn.LayerNorm()(obs_encoding)
            embedding = nn.relu(obs_encoding)
        else:
            obs_encoding = nn.Dense(
                self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
            )(obs)
            obs_encoding = nn.relu(obs_encoding)
            obs_encoding = nn.Dense(
                self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
            )(obs_encoding)
            obs_encoding = nn.LayerNorm()(obs_encoding)
            embedding = nn.relu(obs_encoding)

        if self.memoryless:
            embedding = SimpleNN(hidden_size=self.hidden_size)(embedding)
        else:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)

        actor = Actor(self.action_space, hidden_size=self.hidden_size)
        pi = actor(embedding)

        critic = Critic(hidden_size=self.hidden_size)

        # GVF prediction
        gvf_critic = None
        if self.cumulant_size is not None:
            gvf_critic = GVF(hidden_size=self.hidden_size, out_size=self.cumulant_size)
            if self.double_critic:
                gvf_critic = nn.vmap(GVF,
                                     variable_axes={'params': 0},
                                     split_rngs={'params': True},
                                     in_axes=None,
                                     out_axes=2,
                                     axis_size=2)(hidden_size=self.hidden_size, out_size=self.cumulant_size)

        v = jnp.squeeze(critic(embedding), axis=-1)

        gvf_prediction = None
        if gvf_critic is not None:
            gvf_prediction = gvf_critic(embedding)

        return hidden, pi, v, gvf_prediction, obs_encoding


class CumulantGammaNetwork(nn.Module):
    cumulant_size: int
    gamma: float = 0.9
    gamma_type: str = 'nn_gamma_sigmoid'
    gamma_max: float = 1.
    gamma_min: float = 0.75

    @nn.compact
    def __call__(self, x):
        cumulant_mapped = nn.Dense(features=self.cumulant_size)(x)
        cumulant_mapped = nn.LayerNorm()(cumulant_mapped)

        if self.gamma_type == 'nn_gamma_sigmoid':
            hangman = nn.Dense(features=1)(x)
            hangman = nn.sigmoid(hangman)
            hangman = (self.gamma_max - self.gamma_min) * hangman + self.gamma_min
        elif self.gamma_type == 'fixed':
            hangman = jnp.zeros((x.shape[0], 1)) + self.gamma

        return cumulant_mapped, hangman

