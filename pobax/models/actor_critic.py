from typing import Union

import flax.linen as nn
from gymnax.environments import spaces
from jax._src.nn.initializers import orthogonal, constant
import jax.numpy as jnp

from pobax.models.actor import Actor
from pobax.models.critic import Critic, GVF
from pobax.models.network import SimpleNN, ScannedRNN, FullImageCNN


class ActorCritic(nn.Module):
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

        obs_gvf = None
        critic = Critic(hidden_size=self.hidden_size)
        if self.gvf_type is not None:
            if self.gvf_type == 'obs':
                gvf_hidden_size = obs.shape[-1]
            elif self.gvf_type == 'hidden_state':
                gvf_hidden_size = self.hidden_size

            gvf_critic = GVF(hidden_size=gvf_hidden_size)

        if self.double_critic:
            critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)
            if self.gvf_type is not None:
                gvf_critic = nn.vmap(GVF,
                                     variable_axes={'params': 0},
                                     split_rngs={'params': True},
                                     in_axes=None,
                                     out_axes=2,
                                     axis_size=2)(hidden_size=self.hidden_size)

        v = jnp.squeeze(critic(embedding), axis=-1)
        if gvf_critic is not None:
            if self.gvf_type == 'obs':
                obs_gvf = gvf_critic(obs)
            elif self.gvf_type == 'hidden_state':
                obs_gvf = gvf_critic(embedding)

        return hidden, pi, v, obs_gvf


class GVFActorCritic(nn.Module):
    action_space: Union[spaces.Discrete, spaces.Box]
    hidden_size: int = 128
    gvf_obs: bool = False
    memoryless: bool = False
    double_critic: bool = False

    @nn.compact
    def __call__(self, hidden, x):
        obs, dones = x
        embedding = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
        )(obs)
        embedding = nn.relu(embedding)

        if len(x.shape) > 1:
            if self.memoryless:
                embedding = SimpleNN(hidden_size=self.hidden_size)(embedding)
            else:
                rnn_in = (embedding, dones)
                hidden, embedding = ScannedRNN(hidden_size=self.hidden_size)(hidden, rnn_in)

        actor = Actor(self.action_space, hidden_size=self.hidden_size)
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
