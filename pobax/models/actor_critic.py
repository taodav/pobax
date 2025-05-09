import math
from typing import Union, Any

import flax.linen as nn
import jax.lax
from gymnax.environments import spaces
from jax._src.nn.initializers import orthogonal, constant
import jax.numpy as jnp

from pobax.models.actor import Actor
from pobax.models.critic import Critic, GVF
from pobax.models.network import SimpleNN, ScannedRNN, FullImageCNN


class Encoder(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, obs):
        if len(obs.shape) > 3:
            obs_encoding = FullImageCNN(hidden_size=self.hidden_size)(obs)
        else:
            obs_encoding = nn.Dense(
                self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
            )(obs)
            obs_encoding = nn.LayerNorm()(obs_encoding)
            obs_encoding = nn.tanh(obs_encoding)
            obs_encoding = nn.Dense(
                self.hidden_size, kernel_init=orthogonal(jnp.sqrt(2)), bias_init=constant(0.0)
            )(obs_encoding)

        return obs_encoding


class SFNetwork(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        # Now we do our SF critic
        critic_embedding = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        critic_embedding = nn.relu(critic_embedding)
        critic_embedding = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            critic_embedding
        )
        critic_embedding = nn.relu(critic_embedding)
        critic_embedding = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            critic_embedding
        )
        return critic_embedding

def normalize(arr: jnp.ndarray, p: int = 2, axis: int = 0, eps: float = 1e-12):
    denom = jnp.linalg.norm(arr, ord=p, axis=axis, keepdims=True)
    denom = jnp.clip(denom, min=eps)
    return arr / denom


class SFActorCritic(nn.Module):
    action_space: Union[spaces.Discrete, spaces.Box]
    hidden_size: int = 128
    memoryless: bool = False
    double_critic: bool = False
    n_rewards: int = 1

    def setup(self) -> None:
        self.encoder = Encoder(self.hidden_size)
        if self.memoryless:
            self.rnn = SimpleNN(hidden_size=self.hidden_size)
        else:
            self.rnn = ScannedRNN(hidden_size=self.hidden_size)
        self.actor = Actor(self.action_space, hidden_size=self.hidden_size)
        if self.double_critic:
            self.sf = nn.vmap(SFNetwork,
                              variable_axes={'params': 0},
                              split_rngs={'params': True},
                              in_axes=None,
                              out_axes=2,
                              axis_size=2)(hidden_size=self.hidden_size)
        else:
            self.sf = SFNetwork(hidden_size=self.hidden_size)
        self.reward_fn = nn.Dense(self.n_rewards, name='r', kernel_init=orthogonal(2), bias_init=constant(0.0))

    def __call__(self, hidden, x):
        """
        Does a forward pass, with a stop gradient through
        :param hidden:
        :param x:
        :return:
        """
        obs, dones = x

        encoded_obs = self.encoder(obs)

        if self.memoryless:
            hs = self.rnn(encoded_obs)
        else:
            rnn_in = (encoded_obs, dones)
            hidden, hs = self.rnn(hidden, rnn_in)

        pi = self.actor(hs)

        sf_embedding = self.sf(hs)

        v = self.reward_fn(sf_embedding)

        return hidden, pi, v

    def get_reward(self, encoding):
        return self.reward_fn(normalize(encoding, p=2, axis=-1))

    def get_encoding(self, obs):
        return self.encoder(obs)

    def get_sf(self, hidden, x, rew_params):
        obs, dones = x

        encoded_obs = self.encoder(obs)

        if self.memoryless:
            hs = self.rnn(encoded_obs)
        else:
            rnn_in = (encoded_obs, dones)
            hidden, hs = self.rnn(hidden, rnn_in)

        pi = self.actor(hs)

        sf_embedding = self.sf(hs)

        rew_w, rew_b = rew_params
        v = sf_embedding @ rew_w + rew_b

        return hidden, pi, v, sf_embedding


class ActorCritic(nn.Module):
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


def orthogonal_random_projection() -> jax.nn.initializers.Initializer:
    """
    Builds an initializer that returns random projections of a vector.

    Implemented as per https://en.wikipedia.org/wiki/Random_projection#Orthogonal_random_projection

    Returns:
    An orthogonal initializer.
    """
    def init(key: jnp.ndarray,
           shape: tuple,
           dtype: Any = float) -> jnp.ndarray:
        dtype = jax._src.dtypes.canonicalize_dtype(dtype)
        if len(shape) < 2:
          raise ValueError("orthogonal initializer requires at least a 2D shape")
        bern = jax.random.bernoulli(key, shape=shape).astype(float)
        shifted_bern = bern - (bern == 0).astype(float)
        scaled_shifted_bern = (1 / jnp.sqrt(shape[0])) * shifted_bern.astype(dtype)
        return scaled_shifted_bern
    return init

def sparse_orthogonal_random_projection() -> jax.nn.initializers.Initializer:
    """
    Builds an initializer that returns random projections of a vector.

    Implemented as per https://en.wikipedia.org/wiki/Random_projection#Orthogonal_random_projection

    Returns:
    An orthogonal initializer.
    """
    def init(key: jnp.ndarray,
             shape: tuple,
             dtype: Any = float) -> jnp.ndarray:
        dtype = jax._src.dtypes.canonicalize_dtype(dtype)
        if len(shape) < 2:
            raise ValueError("orthogonal initializer requires at least a 2D shape")
        p = jnp.array([1, 4, 1]) / 6  # we do this to make sure things sum to 1
        multnom = jax.random.choice(key, jnp.array([-1, 0, 1]), p=p, shape=shape).astype(float)
        scaled_multnom = (3 / jnp.sqrt(shape[0])) * multnom.astype(dtype)
        return scaled_multnom
    return init


class CumulantNetwork(nn.Module):
    cumulant_size: int
    @nn.compact
    def __call__(self, x):
        if len(x.shape) > 3:
            # If x is an image, flatten it
            x = x.reshape((*x.shape[:-3], -1))

        cumulant_mapped = nn.Dense(
            self.cumulant_size, kernel_init=sparse_orthogonal_random_projection(), use_bias=False
        )(x)

        # cumulant_mapped = nn.Dense(features=self.cumulant_size)(x)
        # cumulant_mapped = nn.LayerNorm()(cumulant_mapped)
        return cumulant_mapped

class HangmanNetwork(nn.Module):
    gamma: float = 0.9
    gamma_type: str = 'nn_gamma_sigmoid'
    gamma_max: float = 1.
    gamma_min: float = 0.75

    @nn.compact
    def __call__(self, x):
        if len(x.shape) > 3:
            # If x is an image, flatten it
            x = x.reshape((*x.shape[:-3], -1))

        if self.gamma_type == 'nn_gamma_sigmoid':
            hangman = nn.Dense(features=1)(x)
            hangman = nn.sigmoid(hangman)
            hangman = (self.gamma_max - self.gamma_min) * hangman + self.gamma_min
        elif self.gamma_type == 'fixed':
            hangman = jnp.zeros((x.shape[0], 1)) + self.gamma

        return hangman


class RandomRewardNetwork(nn.Module):
    n_rewards: int = 1
    @nn.compact
    def __call__(self, x):
        if len(x.shape) > 3:
            # If x is an image, flatten it
            x = x.reshape((x.shape[:-3], -1))

        rewards = nn.Dense(
            self.n_rewards, kernel_init=sparse_orthogonal_random_projection(), use_bias=False
        )(x)
        return rewards

