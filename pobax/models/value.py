import functools
import math
from typing import Sequence

import chex
import distrax
from gymnax.environments import environment, spaces
import jax
import numpy as np
from flax import linen as nn
from jax import numpy as jnp
from jax._src.nn.initializers import orthogonal, constant


class Critic(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, x):
        critic = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        critic = nn.relu(critic)
        critic = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(
            critic
        )
        return critic

