import functools
import distrax
import flax.linen as nn
import jax
import jax.numpy as jnp
from jax._src.nn.initializers import orthogonal, constant
import numpy as np


class SmallImageCNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        # 5x5
        if x.shape[-3] == x.shape[-2] and x.shape[-3] == 5:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(4, 4), strides=1, padding=1)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.hidden_size, kernel_size=(3, 3), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out2)

        # 3x3
        elif x.shape[-3] == x.shape[-2] and x.shape[-3] == 3:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out1)

        # 10x10
        elif x.shape[-3] == x.shape[-2] and x.shape[-3] == 10:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(5, 5), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.hidden_size, kernel_size=(4, 4), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(3, 3), strides=1, padding=0)(out2)

        elif x.shape[-3] == x.shape[-2] and x.shape[-3] == 64:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=(8, 8), strides=2, padding='SAME')(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.hidden_size, kernel_size=(6, 6), strides=2, padding='SAME')(out1)
            out2 = nn.relu(out2)
            out3 = nn.Conv(features=self.hidden_size, kernel_size=(4, 4), strides=2, padding='SAME')(out2)
            out3 = nn.relu(out3)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(3, 3), strides=2, padding='SAME')(out3)

        else:
            raise NotImplementedError

        conv_out = nn.relu(conv_out)
        # Convolutions "flatten" the last three dimensions.
        flat_out = conv_out.reshape((*conv_out.shape[:-3], -1))  # Flatten
        final_out = nn.Dense(features=self.hidden_size)(flat_out)
        return final_out
    

class Game2048CNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, observation):
        board = observation['board'] # Adjust for input structure
        board = jnp.expand_dims(board, axis=-1)
        # Convolutional layers
        x = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding='VALID')(board)
        x = nn.relu(x)
        x = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding='VALID')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1)(x)
        conv_out = nn.relu(x)
        # flat_out = conv_out.reshape((conv_out.shape[0], -1, conv_out.shape[-1]))
        flat_out = conv_out.reshape((*conv_out.shape[:-3], -1))
        # MLP
        final_out = nn.Dense(features=self.hidden_size)(flat_out)
        return final_out
    
class Game2048Actor(nn.Module):
    action_dim: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x, obs):
        action_mask = obs['action_mask']
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)
        
        masked_logits = jnp.where(
                action_mask, actor_mean, jnp.finfo(jnp.float32).min
            )

        pi = distrax.Categorical(logits=masked_logits)
        return pi
    

class SokobanCNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, observation):
        # Assuming observation['board'] is a matrix representation of the game state
        grid = observation['grid']
        step_count = observation['step_count']
        time_limit = 120
        # board = jnp.expand_dims(board, axis=-1)  # Add channel dimension for CNN compatibility
        x_processed = self.preprocess_input(grid)
        # Convolutional layers
        x = nn.Conv(features=self.hidden_size, kernel_size=(3, 3), strides=1, padding='SAME')(x_processed)
        x = nn.relu(x)
        x = nn.Conv(features=self.hidden_size, kernel_size=(3, 3), strides=1, padding='SAME')(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.hidden_size, kernel_size=(3, 3), strides=1, padding='SAME')(x)
        x = nn.relu(x)

        # Flatten the output for potential use in dense layers
        flat_out = x.reshape((*x.shape[:-3], -1))  # Flatten while keeping the batch and channel dimensions
        
        # MLP - Optional depending on further processing needs
        final_out = nn.Dense(features=self.hidden_size)(flat_out)

        norm_step_count = jnp.expand_dims(step_count / time_limit, axis=-1)
        final_out = jnp.concatenate([final_out, norm_step_count], axis=-1)
        final_out = nn.Dense(features=self.hidden_size)(final_out)
        final_out = nn.relu(final_out)
        return final_out
    
    def preprocess_input(self, 
        input_array: jnp.ndarray,
    ) -> jnp.ndarray:

        one_hot_array_fixed = jnp.equal(input_array[..., 0:1], jnp.array([3, 4])).astype(
            jnp.float32
        )

        one_hot_array_variable = jnp.equal(input_array[..., 1:2], jnp.array([1, 2])).astype(
            jnp.float32
        )

        total = jnp.concatenate((one_hot_array_fixed, one_hot_array_variable), axis=-1)

        return total
    
class SokobanActor(nn.Module):
    action_dim: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x, obs):
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        actor_mean = nn.relu(actor_mean)
        actor_mean = nn.Dense(
            self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(actor_mean)

        pi = distrax.Categorical(logits=actor_mean)
        return pi
    
class SnakeCNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, observation):
        # Assuming observation['grid'] is a matrix representation of the snake game board
        grid = observation['grid']
        step_count = observation['step_count']
        time_limit = 4000
        # grid = jnp.expand_dims(grid, axis=-1)  # Add channel dimension

        # Convolutional layers
        x = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=2, padding='SAME')(grid)
        x = nn.relu(x)
        x = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding='SAME')(x)
        x = nn.relu(x)

        # Flatten the output to prepare for dense layer processing
        flat_out = x.reshape((*x.shape[:-3], -1))  # Flatten preserving batch size
        norm_step_count = jnp.expand_dims(step_count / time_limit, axis=-1)
        final_out = jnp.concatenate([flat_out, norm_step_count], axis=-1)
        final_out = nn.Dense(features=self.hidden_size)(final_out)
        final_out = nn.relu(final_out)
        return final_out

class SnakeActor(nn.Module):
    action_dim: int
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x, obs):
        # Retrieve the action mask from the observation
        action_mask = obs['action_mask']

        # Dense layers for action logits
        actor_output = nn.Dense(self.hidden_size)(x)
        actor_output = nn.relu(actor_output)
        # actor_output = nn.Dense(self.hidden_size)(x)
        # actor_output = nn.relu(actor_output)
        actor_output = nn.Dense(self.action_dim)(actor_output)

        # Apply the action mask to the logits
        masked_logits = jnp.where(
            action_mask, actor_output, jnp.finfo(jnp.float32).min
        )

        # Create a categorical distribution for action selection
        pi = distrax.Categorical(logits=masked_logits)
        return pi