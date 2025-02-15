import distrax
import flax.linen as nn
import jax.numpy as jnp
from jax._src.nn.initializers import orthogonal, constant
import numpy as np

from . import FullImageCNN
from .network import SimpleNN, ScannedRNN, SmallImageCNN
from .value import Critic
from pobax.models.transformerXL import Transformer


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


class DiscreteActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, _, x):
        obs, dones = x

        embedding = SimpleNN(hidden_size=self.hidden_size)(obs)

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

        v = critic(embedding)

        return hidden, pi, jnp.squeeze(v, axis=-1)


class ImageDiscreteActorCritic(nn.Module):
    action_dim: int
    hidden_size: int = 128
    double_critic: bool = False

    @nn.compact
    def __call__(self, _, x):
        obs, dones = x

        # if obs.shape[-2] >= 64:
        embedding = FullImageCNN(hidden_size=self.hidden_size)(obs)
        # else:
        #     embedding = SmallImageCNN(hidden_size=self.hidden_size)(obs)
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

        # if obs.shape[-2] >= 64:
        embedding = FullImageCNN(hidden_size=self.hidden_size)(obs)
        # else:
        #     embedding = SmallImageCNN(hidden_size=self.hidden_size)(obs)
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

        # if we're in this case, obs is an image
        if len(obs.shape) == 4:
            valid_action_mask = (obs == 0).reshape(*obs.shape[:-2], -1)
            embedding = SmallImageCNN(hidden_size=self.hidden_size)(obs)
            embedding = nn.relu(embedding)
        else:
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

        # if we're in this case, obs is an image
        if len(obs.shape) == 4:
            valid_action_mask = (obs == 0).reshape(*obs.shape[:-2], -1)
            embedding = SmallImageCNN(hidden_size=self.hidden_size)(obs)
            embedding = nn.relu(embedding)
        else:
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


class DiscreteActorCriticTransformer(nn.Module):
    action_dim: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers:int
    hidden_size: int = 128
    gating:bool=False
    gating_bias:float=0.
    double_critic: bool = False

    @nn.compact
    def __call__(self, memories,obs,mask):
        print('obs', obs.shape)
        transformer = Transformer(
                                encoder_size=self.encoder_size,
                                num_heads=self.num_heads,
                                qkv_features=self.qkv_features,
                                num_layers=self.num_layers,gating=self.gating,gating_bias=self.gating_bias)
        embedding, memory_out = transformer(memories,obs,mask)
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
        return pi, jnp.squeeze(v, axis=-1),memory_out
    
    @nn.compact
    def model_forward_eval(self, memories,obs,mask):
        """Used during environment rollout (single timestep of obs). And return the memory"""
        transformer = Transformer(
                                encoder_size=self.encoder_size,
                                num_heads=self.num_heads,
                                qkv_features=self.qkv_features,
                                num_layers=self.num_layers,gating=self.gating,gating_bias=self.gating_bias)
        embedding,memory_out = transformer.forward_eval(memories,obs,mask)

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

        return pi, jnp.squeeze(v, axis=-1),memory_out
    
    @nn.compact
    def model_forward_train(self, memories,obs,mask): 
        """Used during training: a window of observation is sent. And don't return the memory"""
        transformer = Transformer(
                                encoder_size=self.encoder_size,
                                num_heads=self.num_heads,
                                qkv_features=self.qkv_features,
                                num_layers=self.num_layers,gating=self.gating,gating_bias=self.gating_bias)
        embedding = transformer.forward_train(memories,obs,mask)

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
        return pi, jnp.squeeze(v, axis=-1)
    

class ImageDiscreteActorCriticTransformer(nn.Module):
    action_dim: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers:int
    hidden_size: int = 128
    gating:bool=False
    gating_bias:float=0.
    double_critic: bool = False
    
    @nn.compact
    def __call__(self, memories,obs,mask):
        embedding = FullImageCNN(hidden_size=self.hidden_size)(obs)
        embedding = nn.relu(embedding)
        embedding = embedding.squeeze(1)

        transformer = Transformer(
                                encoder_size=self.encoder_size,
                                num_heads=self.num_heads,
                                qkv_features=self.qkv_features,
                                num_layers=self.num_layers,gating=self.gating,gating_bias=self.gating_bias)
        embedding, memory_out = transformer(memories,embedding,mask)
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
        return pi, jnp.squeeze(v, axis=-1),memory_out
    
    @nn.compact
    def model_forward_eval(self, memories,obs,mask):
        """Used during environment rollout (single timestep of obs). And return the memory"""
        embedding = FullImageCNN(hidden_size=self.hidden_size)(obs)
        embedding = nn.relu(embedding)
        embedding = embedding.squeeze(1)

        transformer = Transformer(
                                encoder_size=self.encoder_size,
                                num_heads=self.num_heads,
                                qkv_features=self.qkv_features,
                                num_layers=self.num_layers,gating=self.gating,gating_bias=self.gating_bias)
        embedding,memory_out = transformer.forward_eval(memories,embedding,mask)

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

        return pi, jnp.squeeze(v, axis=-1),memory_out
    
    @nn.compact
    def model_forward_train(self, memories,obs,mask): 
        """Used during training: a window of observation is sent. And don't return the memory"""
        embedding = FullImageCNN(hidden_size=self.hidden_size)(obs)
        embedding = nn.relu(embedding)

        transformer = Transformer(
                                encoder_size=self.encoder_size,
                                num_heads=self.num_heads,
                                qkv_features=self.qkv_features,
                                num_layers=self.num_layers,gating=self.gating,gating_bias=self.gating_bias)
        embedding = transformer.forward_train(memories,embedding,mask)

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
        return pi, jnp.squeeze(v, axis=-1)