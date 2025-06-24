import flax.linen as nn
from jax._src.nn.initializers import orthogonal, constant

class CNN(nn.Module):
    hidden_size: int
    num_channels: int = 32

    @nn.compact
    def __call__(self, x):
        if len(x.shape) == 4:
            num_dims = 3
        else:
            num_dims = len(x.shape) - 2  # b x num_envs
        # 10x10 2 dimensions
        if num_dims == 2 and x.shape[-2] == x.shape[-1] and x.shape[-2] == 10:
            out1 = nn.Conv(features=self.hidden_size, kernel_size=5, strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.hidden_size, kernel_size=4, strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=3, strides=1, padding=0)(out2)

        # 5x5
        elif x.shape[-3] == x.shape[-2] and x.shape[-3] == 5:
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

        elif x.shape[-2] == 7 and x.shape[-3] == 4:
            out1 = nn.Conv(features=64, kernel_size=(2, 4), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=128, kernel_size=(2, 3), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)
            conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out2)
        elif x.shape[-2] == 5 and x.shape[-3] == 3:
            out1 = nn.Conv(features=64, kernel_size=(2, 3), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            conv_out = nn.Conv(features=128, kernel_size=(2, 2), strides=1, padding=0)(out1)
            # out2 = nn.relu(out2)
            # conv_out = nn.Conv(features=self.hidden_size, kernel_size=(2, 2), strides=1, padding=0)(out2)

        elif x.shape[-2] == 3 and x.shape[-3] == 2:
            out1 = nn.Conv(features=64, kernel_size=(1, 1), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            conv_out = nn.Conv(features=128, kernel_size=(2, 2), strides=1, padding=0)(out1)

        elif x.shape[-2] >= 14 and x.shape[-2] <= 64:
            out1 = nn.Conv(features=64, kernel_size=(6, 6), strides=1, padding=0)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=64, kernel_size=(5, 5), strides=1, padding=0)(out1)
            out2 = nn.relu(out2)

            final_out = out2
            conv_out = nn.Conv(features=64, kernel_size=(2, 2), strides=1, padding=0)(final_out)

        else:
            out1 = nn.Conv(features=self.num_channels, kernel_size=(7, 7), strides=4)(x)
            out1 = nn.relu(out1)
            out2 = nn.Conv(features=self.num_channels, kernel_size=(5, 5), strides=2)(out1)
            out2 = nn.relu(out2)
            out3 = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=2)(out2)
            out3 = nn.relu(out3)
            conv_out = nn.Conv(features=self.num_channels, kernel_size=(3, 3), strides=2)(out3)
            
        conv_out = nn.relu(conv_out)
        # Convolutions "flatten" the last num_dims dimensions.
        flat_out = conv_out.reshape((*conv_out.shape[:-num_dims], -1))  # Flatten
        final_out = nn.Dense(features=self.hidden_size)(flat_out)
        return final_out


class SimpleNN(nn.Module):
    hidden_size: int

    @nn.compact
    def __call__(self, x):
        out = nn.Dense(self.hidden_size, kernel_init=orthogonal(2), bias_init=constant(0.0))(
            x
        )
        out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        out = nn.relu(out)
        out = nn.Dense(
            self.hidden_size, kernel_init=orthogonal(0.01), bias_init=constant(0.0)
        )(out)
        return out

class BattleshipEmbedding(nn.Module):
    hidden_size: int = 128

    @nn.compact
    def __call__(self, obs):
        hit = obs[..., 0:1]
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
        return embedding