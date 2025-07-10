import distrax
import flax.linen as nn
import jax.numpy as jnp
from jax._src.nn.initializers import orthogonal, constant
import numpy as np
from pobax.models.transformerXL import Transformer
from .network import SimpleNN, ScannedRNN, SmallImageCNN, FullImageCNN
from .value import Critic


class ContinuousActor(nn.Module):
    action_dim: int
    hidden_size: int = 128
    activation: str = "tanh"

    @nn.compact
    def __call__(self, x, action_mask=None):
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
        # TODO: add action mask
        if action_mask is not None:
            # TODO: implement action mask
            raise NotImplementedError("Action mask is not implemented yet")
        pi = distrax.MultivariateNormalDiag(actor_mean, jnp.exp(actor_logtstd))
        return pi

class ContinuousActorCriticTransformer(nn.Module):
    action_dim: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers:int
    hidden_size: int = 128
    gating:bool=False
    gating_bias:float=0.
    activation: str = "tanh"
    double_critic: bool = False

    def setup(self):
        self.transformer = Transformer(
            encoder_size=self.encoder_size,
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            num_layers=self.num_layers,
            gating=self.gating,
            gating_bias=self.gating_bias
        )
        self.actor = ContinuousActor(self.action_dim, hidden_size=self.hidden_size,
                                activation=self.activation)
        if self.double_critic:
            self.critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)
        else:
            self.critic = Critic(hidden_size=self.hidden_size)
    
    def __call__(self, memories,obs,mask):
        embedding, memory_out = self.transformer(memories,obs,mask)
        pi = self.actor(embedding)
        v = self.critic(embedding)
        return pi, jnp.squeeze(v, axis=-1),memory_out
    
    def model_forward_eval(self, memories,obs,mask):
        """Used during environment rollout (single timestep of obs). And return the memory"""
        embedding,memory_out = self.transformer.forward_eval(memories,obs,mask)
        pi = self.actor(embedding)

        v = self.critic(embedding)

        return pi, jnp.squeeze(v, axis=-1),memory_out
    
    def model_forward_train(self, memories,obs,mask): 
        """Used during training: a window of observation is sent. And don't return the memory"""
        embedding = self.transformer.forward_train(memories,obs,mask)
        pi = self.actor(embedding)
        v = self.critic(embedding)
        return pi, jnp.squeeze(v, axis=-1)
    

class ImageContinuousActorCriticTransformer(nn.Module):
    action_dim: int
    encoder_size: int
    num_heads: int
    qkv_features: int
    num_layers:int
    hidden_size: int = 128
    activation: str = "tanh"
    gating:bool=False
    gating_bias:float=0.
    double_critic: bool = False

    def setup(self):
        self.cnn_full = FullImageCNN(hidden_size=self.hidden_size)
        self.cnn_small = SmallImageCNN(hidden_size=self.hidden_size)
        self.transformer = Transformer(
            encoder_size=self.encoder_size,
            num_heads=self.num_heads,
            qkv_features=self.qkv_features,
            num_layers=self.num_layers,
            gating=self.gating,
            gating_bias=self.gating_bias
        )
        self.actor = ContinuousActor(self.action_dim, hidden_size=self.hidden_size,
                                activation=self.activation)
        if self.double_critic:
            self.critic = nn.vmap(Critic,
                             variable_axes={'params': 0},
                             split_rngs={'params': True},
                             in_axes=None,
                             out_axes=2,
                             axis_size=2)(hidden_size=self.hidden_size)
        else:
            self.critic = Critic(hidden_size=self.hidden_size)
    
    def __call__(self, memories,obs,mask):
        if obs.shape[-2] >= 20:
            embedding = self.cnn_full(obs)
        else:
            embedding = self.cnn_small(obs)
        embedding = nn.relu(embedding)
        
        embedding, memory_out = self.transformer(memories,embedding,mask)
        pi = self.actor(embedding)
        v = self.critic(embedding)
        return pi, jnp.squeeze(v, axis=-1),memory_out
    
    def model_forward_eval(self, memories,obs,mask):
        """Used during environment rollout (single timestep of obs). And return the memory"""
        if obs.shape[-2] >= 20:
            embedding = self.cnn_full(obs)
        else:
            embedding = self.cnn_small(obs)
        embedding = nn.relu(embedding)
        
        embedding,memory_out = self.transformer.forward_eval(memories,embedding,mask)
        pi = self.actor(embedding)

        v = self.critic(embedding)

        return pi, jnp.squeeze(v, axis=-1),memory_out
    
    def model_forward_train(self, memories,obs,mask): 
        """Used during training: a window of observation is sent. And don't return the memory"""
        if obs.shape[-2] >= 20:
            embedding = self.cnn_full(obs)
        else:
            embedding = self.cnn_small(obs)
        embedding = nn.relu(embedding)
        
        embedding = self.transformer.forward_train(memories,embedding,mask)

        pi = self.actor(embedding)

        v = self.critic(embedding)
        return pi, jnp.squeeze(v, axis=-1)