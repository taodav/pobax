from functools import partial

import chex
import jax
import jax.numpy as jnp
from jax import random, lax
import gymnax
from gymnax.environments.environment import Environment, EnvParams


@chex.dataclass
class BattleShipState:
    hits_misses: jnp.ndarray  # rows x cols, 0 = never hit, 1 = miss, 2 = hit
    board: jnp.ndarray  # rows x cols


@partial(jax.jit, static_argnames=['ship_length'])
def place_ship_randomly(key: chex.PRNGKey, board: jnp.ndarray, ship_length: int):
    def check_vert_ship(y: int, x: int):
        return jnp.all(lax.dynamic_slice(board, (y, x), (ship_length, 1)) == 0)

    def check_horz_ship(y: int, x: int):
        return jnp.all(lax.dynamic_slice(board, (y, x), (1, ship_length)) == 0)

    vert_y_max = board.shape[1] - ship_length + 1
    horz_x_max = board.shape[0] - ship_length + 1

    check_all_vert = jax.vmap(jax.vmap(check_vert_ship, in_axes=[None, 0]), in_axes=[0, None])
    check_all_horz = jax.vmap(jax.vmap(check_horz_ship, in_axes=[None, 0]), in_axes=[0, None])

    checks_vert = check_all_vert(jnp.arange(vert_y_max), jnp.arange(board.shape[1]))
    checks_horz = check_all_horz(jnp.arange(board.shape[0]), jnp.arange(horz_x_max))

    pose_key, choice_key = random.split(key)
    pose = random.bernoulli(pose_key)

    def sample_true_position(key, matrix):
        # Flatten the matrix
        flat_matrix = matrix.ravel()

        # Create log probabilities: -inf for False (0) values, 0 for True (1) values
        log_probs = jnp.where(flat_matrix, 0, -jnp.inf)

        # Sample a flat index based on these log probabilities
        sampled_index = random.categorical(key, log_probs, shape=())

        # Convert flat index back to 2D index
        return jnp.divmod(sampled_index, matrix.shape[1])

    sampled_vert_pos = sample_true_position(choice_key, checks_vert)
    sampled_horz_pos = sample_true_position(choice_key, checks_horz)

    board_updated_vert = lax.dynamic_update_slice(board, jnp.ones(ship_length, dtype=int)[..., None], sampled_vert_pos)
    board_updated_horz = lax.dynamic_update_slice(board, jnp.ones(ship_length, dtype=int)[None, ...], sampled_horz_pos)

    new_pos, new_board = lax.cond(pose,
                                  lambda : (sampled_horz_pos, board_updated_horz),
                                  lambda : (sampled_vert_pos, board_updated_vert))

    return new_board, new_pos, pose.astype(int)


class Battleship(Environment):
    def __init__(self,
                 rows: int = 10,
                 cols: int = 10,
                 ship_lengths: tuple[int] = (5, 4, 3, 2),
                 dense_reward: bool = False):
        self.rows = rows
        self.cols = cols
        self.ship_lengths = ship_lengths
        self.dense_reward = dense_reward

        # We set gamma to 1, since we're in the finite horizon case.
        self.gamma = 1.

    def action_space(self, params: EnvParams):
        return gymnax.environments.spaces.Discrete(self.rows * self.cols)

    def observation_space(self, params: EnvParams):
        """
        Obs space is whether or not you hit a ship +
        action_mask for illegal actions.
        """
        return gymnax.environments.spaces.Box(0, 1, (1 + self.action_space(params).n,))

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)

    @partial(jax.jit, static_argnums=(0,))
    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
    ) -> tuple[chex.Array, BattleShipState]:
        board = jnp.zeros((self.rows, self.cols), dtype=int)
        positions = jnp.zeros((len(self.ship_lengths), 2), dtype=int)
        poses = jnp.zeros(len(self.ship_lengths), dtype=int)

        # We HAVE to do a for loop here, since ship lengths are not the same.
        for i, l in enumerate(self.ship_lengths):
            place_key, key = random.split(key)
            board, pos, pose = place_ship_randomly(place_key, board, l)
            positions = positions.at[i].set(pos)
            poses = poses.at[i].set(pose)
        state = BattleShipState(
            hits_misses=jnp.zeros_like(board, dtype=int),
            board=board
        )
        return self.get_obs(state, state, params), state

    def get_obs(
        self,
        prev_state: BattleShipState,
        state: BattleShipState,
        params: EnvParams,
    ) -> chex.Array:
        valid_actions = state.hits_misses == 0
        valid_actions = valid_actions.flatten()
        hit = (prev_state.hits_misses == 2).sum() < (state.hits_misses == 2).sum()
        return jnp.concatenate((jnp.array([hit], dtype=float), valid_actions), axis=-1)

    @partial(jax.jit, static_argnums=(0,))
    def step_env(
        self,
        key: chex.PRNGKey,
        state: BattleShipState,
        action: int,
        params: EnvParams,
    ) -> tuple[chex.Array, BattleShipState, jnp.ndarray, jnp.ndarray, dict]:
        idx_hit = jnp.unravel_index(action, (self.rows, self.cols))
        targeted_a_ship = state.board[idx_hit]

        # miss idx is 2
        new_val = targeted_a_ship * 2 + (1 - targeted_a_ship)
        new_hits_misses = state.hits_misses.at[idx_hit].set(new_val)
        new_state = state.replace(hits_misses=new_hits_misses.astype(int))
        new_obs = self.get_obs(state, new_state, params)

        done = jnp.all(
            jnp.logical_or(
                jnp.logical_and((new_state.hits_misses == 2), new_state.board),
                jnp.logical_not(new_state.board))
        )
        if self.dense_reward:
            # 0th index obs is hit or not
            reward = new_obs[0]
        else:
            # reward = (self.rows * self.cols) * done - (1 - done)
            reward = (self.rows * self.cols) * done - (1 - done)

        return (
            lax.stop_gradient(new_obs),
            lax.stop_gradient(new_state),
            reward,
            done,
            {}
        )


# class HitsMissBattleship(BattleShip)



