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
    gvf_type: str = None  # [obs, hidden_state, None]
    memoryless: bool = False
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x

        if len(x.shape) > 1:
            embedding = FullImageCNN(hidden_size=self.hidden_size)(obs)
        else:
            embedding = nn.Dense(
                self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
            )(obs)
            embedding = nn.relu(embedding)

        if self.memoryless:
            embedding = SimpleNN(hidden_size=self.hidden_size)(embedding)
        else:
            rnn_in = (embedding, dones)
            hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)

        actor = Actor(self.action_space, hidden_size=self.hidden_size)
        pi = actor(embedding)

        critic = Critic(hidden_size=self.hidden_size)
        if self.gvf_type is not None:
            if self.gvf_type == 'obs':
                gvf_out_size = obs.shape[-1]
            elif self.gvf_type == 'hidden_state':
                gvf_out_size = self.hidden_size

            gvf_critic = GVF(hidden_size=self.hidden_size, out_size=gvf_out_size)

        if self.double_critic and self.gvf_type is not None:
            # critic = nn.vmap(Critic,
            #                  variable_axes={'params': 0},
            #                  split_rngs={'params': True},
            #                  in_axes=None,
            #                  out_axes=2,
            #                  axis_size=2)(hidden_size=self.hidden_size)
            gvf_critic = nn.vmap(GVF,
                                 variable_axes={'params': 0},
                                 split_rngs={'params': True},
                                 in_axes=None,
                                 out_axes=2,
                                 axis_size=2)(hidden_size=self.hidden_size, out_size=gvf_out_size)

        v = jnp.squeeze(critic(embedding), axis=-1)

        obs_gvf = None
        if gvf_critic is not None:
            if self.gvf_type == 'obs':
                obs_gvf = gvf_critic(obs)
            elif self.gvf_type == 'hidden_state':
                obs_gvf = gvf_critic(embedding)

        return hidden, pi, v, obs_gvf


class GammaOffset(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        if len(x.shape) > 1:
            gamma_offset = FullImageCNN(hidden_size=self.hidden_size // 4, num_channels=4)(x)
            gamma_offset = nn.Dense(features=1)(gamma_offset)
            gamma_offset = nn.tanh(gamma_offset)
        else:
            gamma_offset = SimpleNN(hidden_size=self.hidden_size // 4)(x)
            gamma_offset = nn.Dense(features=1)(gamma_offset)
            gamma_offset = nn.tanh(gamma_offset)
        return gamma_offset

