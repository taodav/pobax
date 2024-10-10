
import chex
import jax.numpy as jnp
from gymnax.environments.misc.reacher import Reacher, EnvState, EnvParams


class ReacherPOMDP(Reacher):
    def get_obs(self, state: EnvState, params: EnvParams, key=None) -> chex.Array:
        """Concatenate reward, one-hot action and time stamp."""
        goal_obs = jnp.reshape(state.goal_xy, (1, 2))
        is_start = jnp.isclose(state.time, 0)
        ob = (
            jnp.reshape(jnp.cos(state.angles), (1, self.num_joints)),
            jnp.reshape(jnp.sin(state.angles), (1, self.num_joints)),
            jnp.reshape(state.angle_vels, (1, self.num_joints)),
            goal_obs * is_start,
        )
        ob = jnp.concatenate(ob, axis=0)
        return jnp.reshape(ob, (-1,))
