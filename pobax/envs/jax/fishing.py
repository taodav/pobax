from functools import partial
import json
from pathlib import Path

import chex
from gymnax.environments import environment, spaces
from gymnax.environments import EnvParams
import jax
from jax import random, lax
import jax.numpy as jnp

from definitions import ROOT_DIR


def positional_choice(key, a, p):
    return jax.random.choice(key, a, p=p)


# Helpers for one-hot encoding
def ind_to_one_hot(arr: jnp.ndarray, max_val: int, channels_first: bool = False) -> jnp.ndarray:
    one_hot = (jnp.arange(max_val + 1) == arr[..., None]).astype(jnp.uint8)
    if channels_first:
        one_hot = jnp.transpose(one_hot, (len(one_hot.shape) - 1, *range(len(one_hot.shape) - 1)))
    return one_hot


def pos_to_map(pos: jnp.ndarray, size: int):
    pos_map = jnp.zeros((size, size), dtype=jnp.int16)
    if len(pos.shape) == 2:
        for p in pos:
            if jnp.all(p > -1):
                pos_map[p[0], p[1]] = 1
    elif len(pos.shape) == 1:
        if jnp.all(pos > -1):
            pos_map[pos[0], pos[1]] = 1
    return pos_map


def parse_currents(currents_list: list) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    In a config file, parses the 'currents' attribute into two arrays.
    One is n_currents size array of the positions of each current.
    The second is an n_currents x (1 + n_directions) size matrix,
    which is a row stochastic matrix of probabilities of each direction, or
    (for the first probability) keeping each current the same.
    """
    positions = []
    probs = []

    for c in currents_list:
        num_directions = len(c['mapping'])
        for pos in c['mapping']:
            positions.append(pos)
        cr = c['change_rate']
        if cr > 0:
            lamb = 1 / c['change_rate']
            change_prob = lamb * jnp.exp(-lamb)
        else:
            change_prob = 1
        per_direction_prob = change_prob / len(c['directions'])

        direction_prob = jnp.zeros(4)
        direction_prob = direction_prob.at[jnp.array(c['directions'], dtype=int)].set(per_direction_prob)

        prob = jnp.concatenate([jnp.array([1 - change_prob]), direction_prob])
        probs += [prob] * num_directions
    return jnp.array(positions), jnp.array(probs)


def parse_rewards(rewards_list: list) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    In a config file, parses the 'rewards' attribute into two arrays.
    One is for the position of the reward, the other is for the
    prob of resetting (1) the reward.
    """
    positions = []
    probs = []

    for r in rewards_list:
        positions.append(r['position'])
        lamb = 1 / r['change_rate'] if r['change_rate'] > 0 else 0
        change_prob = lamb * jnp.exp(-lamb)
        probs.append(change_prob)

    return jnp.array(positions), jnp.array(probs)


def get_occlusion_mask(obstacle_map: jnp.ndarray, glass_map: jnp.ndarray) -> jnp.ndarray:
    """
    Given an agent-centric obstacle map and an agent-centric glass map,
    calculate an occlusion mask.
    """
    middle = obstacle_map.shape[0] // 2
    curr_occlusion_mask = jnp.zeros_like(obstacle_map)

    def get_occlusion_row(obstacle_row: jnp.ndarray, glass_row: jnp.ndarray) -> jnp.ndarray:
        # TODO: ONLY WORKS FOR WINDOW SIZE 5 NOW
        assert obstacle_row.shape[0] == 5
        not_glass_row = 1 - glass_row
        og1, ogm2 = obstacle_row[1] * not_glass_row[1], obstacle_row[-2] * not_glass_row[-2]
        og2, ogm3 = obstacle_row[2] * not_glass_row[2], obstacle_row[-3] * not_glass_row[-3]
        og_middle = obstacle_row[2] * not_glass_row[2]
        next_row_occlusion = jnp.array([og1, og1 * og2, og_middle, ogm3 * ogm2, ogm2]).squeeze()

        return next_row_occlusion

    vmapped_get_occlusion_row = jax.vmap(get_occlusion_row, in_axes=0)
    new_occlusion_row = jnp.maximum(vmapped_get_occlusion_row(obstacle_map[1:middle], glass_map[1:middle]),
                                    curr_occlusion_mask[0:middle - 1])
    curr_occlusion_mask = curr_occlusion_mask.at[0:middle - 1].set(new_occlusion_row)

    def rot_and_set_occlusion(maps: jnp.ndarray, _):
        curr_occlusion_mask, curr_obstacle_map, curr_glass_map = maps
        curr_occlusion_mask = jnp.rot90(curr_occlusion_mask)
        curr_obstacle_map = jnp.rot90(curr_obstacle_map)
        curr_glass_map = jnp.rot90(curr_glass_map)
        new_occlusion_row = jnp.maximum(vmapped_get_occlusion_row(curr_obstacle_map[1:middle], curr_glass_map[1:middle]),
                                        curr_occlusion_mask[0:middle - 1])
        curr_occlusion_mask = curr_occlusion_mask.at[0:middle - 1].set(new_occlusion_row)
        return (curr_occlusion_mask, curr_obstacle_map, curr_glass_map), _

    maps, _ = jax.lax.scan(rot_and_set_occlusion, (curr_occlusion_mask, obstacle_map, glass_map), None, 3)
    curr_occlusion_mask, _, _ = maps
    curr_occlusion_mask = jnp.rot90(curr_occlusion_mask)

    return curr_occlusion_mask.astype(bool)


@chex.dataclass
class FishingState:
    current_directions: chex.Array
    position: chex.Array
    rewards_active: chex.Array
    steps: int


class Fishing(environment.Environment):
    """
    Fishing environment. We have quite a few more state variables to deal with here,
    So we keep this "epistemic state" environment in group_info.

    In currents, for each group of currents, we have the following key-value mapping:
    mapping: For this group, where are the currents located?
    refresh_rate: On average, how many timesteps until the current flips?
    directions: Directions in which we could sample from for this group.

    Note: this is currently the fully observable version of this environment.
    For partial observability, use the pertaining wrapper
    """
    direction_mapping = jnp.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=int)

    def __init__(self,
                 window_size: int = 5,
                 config_path: Path = Path(ROOT_DIR, 'porl', 'envs', '../configs', 'ocean_nav', 'fishing_8_config.json')
                 ):
        super(Fishing, self).__init__()
        self.config_path = config_path
        self.window_size = window_size

        with open(config_path, 'rb') as f:
            config = json.load(f)
        self.size = config['size']
        self.kelp_prob = config['kelp_prob']
        self.slip_prob = config['slip_prob'] if hasattr(config, 'slip_prob') else 0.
        self.current_positions, self.current_probs = parse_currents(config['currents'])
        self.current_reset_probs = self.current_probs[:, 1:] / self.current_probs[:, 1:].sum(axis=-1, keepdims=True)
        self.reward_positions, self.reward_probs = parse_rewards(config['rewards'])
        self.current_bump_reward = config['current_bump_reward']

        self.kelp_map = jnp.array(config['kelp_map'], dtype=int)
        self.glass_map = jnp.array(config['glass_map'], dtype=int)
        self.obstacle_map = jnp.array(config['obstacle_map'], dtype=int)
        self.starts = jnp.array(config['starts'], dtype=int)

        self.position_max = jnp.array([self.size - 1, self.size - 1], dtype=int)
        self.position_min = jnp.array([0, 0], dtype=int)
        self.vmapped_choice = jax.vmap(positional_choice, in_axes=[0, None, 0])

    @property
    def default_params(self) -> EnvParams:
        return EnvParams(max_steps_in_episode=1000)

    def observation_space(self, params: EnvParams):
        shape = (self.window_size, self.window_size, 6)
        low = jnp.zeros(shape, dtype=int)
        high = jnp.ones(shape, dtype=int)
        return spaces.Box(
            low=low, high=high, shape=shape
        )

    def action_space(self, params: EnvParams):
        return spaces.Discrete(4)

    def reset_currents(self, key: chex.PRNGKey):

        keys = random.split(key, self.current_reset_probs.shape[0])
        current_directions = self.vmapped_choice(keys, self.current_reset_probs.shape[-1], self.current_reset_probs)

        return current_directions

    def reset_env(
        self, key: chex.PRNGKey, params: EnvParams
        ) -> tuple[chex.Array, FishingState]:
        current_key, position_key, reward_key = random.split(key, 3)
        current_directions = self.reset_currents(current_key)
        position = self.starts[random.choice(position_key, self.starts.shape[0])]
        rewards_active = jnp.ones(self.reward_positions.shape[0])
        state = FishingState(current_directions=current_directions,
                             position=position,
                             rewards_active=rewards_active,
                             steps=0)

        return self.get_obs(state, params), state

    def get_full_obs_less_position(self, state: FishingState):
        """
        Returns the full gridworld state, in the shape of a
        self.size x self.size x 1 + 4 + 1 tensor.
        (1st 1 = obstacle map, 2nd 4 = current directions, 3rd 1 = reward map)
        Leaves out the position!
        """
        current_map = jnp.zeros((self.size, self.size, 4))
        current_map = current_map.at[self.current_positions[:, 0],
                                     self.current_positions[:, 1],
                                     state.current_directions].set(1)
        reward_map = jnp.zeros((self.size, self.size))
        reward_map = reward_map.at[self.reward_positions[:, 0],
                                   self.reward_positions[:, 1]].set(state.rewards_active)
        reward_map = reward_map[..., None]
        return jnp.concatenate([self.obstacle_map[..., None], current_map, reward_map], axis=-1)

    def get_obs(
        self,
        state: FishingState,
        params: EnvParams,
    ) -> chex.Array:
        """
        Observation in this case is 3D array:
        1st dimension is channels (we have 7 currently)
        2nd and 3rd is width x height (size x size).
        Channels are:

        obstacle map
        (4x) current maps
        reward map
        """
        unexpanded_map_no_pos = self.get_full_obs_less_position(state)

        # Now we get our expanded map, with our agent at the center.
        expanded_map_size = self.size + self.size - 1
        expanded_agent_pos = expanded_map_size // 2

        expanded_map = jnp.zeros((expanded_map_size, expanded_map_size, unexpanded_map_no_pos.shape[-1]))
        y_start, x_start = expanded_agent_pos - state.position[0], expanded_agent_pos - state.position[1]
        expanded_map = lax.dynamic_update_slice(expanded_map, unexpanded_map_no_pos, (y_start, x_start, 0))

        # Now we expand the glass map.
        expanded_glass = jnp.zeros((expanded_map_size, expanded_map_size), dtype=int)
        expanded_glass = lax.dynamic_update_slice(expanded_glass, self.glass_map, (y_start, x_start))

        # now we shrink to window_size.
        shrank_agent_pos = self.window_size // 2
        shrank_y_start = shrank_x_start = expanded_agent_pos - shrank_agent_pos
        ac_see_thru_map = lax.dynamic_slice(expanded_map, (shrank_y_start, shrank_x_start, 0),
                                            (self.window_size, self.window_size, expanded_map.shape[-1]))

        ac_obstacle_map = ac_see_thru_map[:, :, 0]
        ac_glass_map = lax.dynamic_slice(expanded_glass, (shrank_y_start, shrank_x_start),
                                         (self.window_size, self.window_size))

        occlusion_mask = get_occlusion_mask(ac_obstacle_map, ac_glass_map)

        # Set occluded things to 1 for obstaacles
        final_map = ac_see_thru_map.at[:, :, 0].set(occlusion_mask + (1 - occlusion_mask) * ac_see_thru_map[:, :, 0])

        # Set occluded things to 0 for rewards and currents
        final_map = final_map.at[:, :, 1:5].set((1 - occlusion_mask)[..., None] * ac_see_thru_map[:, :, 1:5])
        final_map = final_map.at[:, :, 5].set((1 - occlusion_mask) * ac_see_thru_map[:, :, 5])

        return final_map

    def get_terminal(self, state: FishingState, params: EnvParams) -> bool:
        return state.steps == params.max_steps_in_episode

    @partial(jax.jit, static_argnums=[0, -1])
    def get_current_reward(self, key: chex.PRNGKey,
                           state: FishingState,
                           prev_state: FishingState,
                           action: int) -> float:
        # Here we see if a current has pushed us into a wall
        position_current_match = jnp.all(state.position[None, ...] == self.current_positions, axis=-1)

        # if we're in a current
        in_current = jnp.any(position_current_match)

        pushed_direction = jnp.dot(position_current_match, state.current_directions).astype(int)
        post_push_position = self.move(key, state.position, pushed_direction, slippage=False)
        current_bump = jnp.all(post_push_position == state.position)
        reward = in_current * current_bump * self.current_bump_reward

        return reward

    @partial(jax.jit, static_argnums=[0, -1])
    def get_reward(self, key: chex.PRNGKey,
                   state: FishingState,
                   prev_state: FishingState,
                   action: int) -> jnp.ndarray:
        reward = ((prev_state.rewards_active == 1) & (state.rewards_active == 0)).sum()

        rew_key = random.split(key)
        reward += self.get_current_reward(rew_key, state, prev_state, action)

        return reward

    @partial(jax.jit, static_argnames=['self', 'action', 'slippage'])
    def move(self, key: chex.PRNGKey, pos: jnp.ndarray,
             action: int, slippage: bool = True):
        if slippage:
            kelp_key, normal_key = random.split(key)
            not_kelp_slip = not_normal_slip = 1
            if self.kelp_prob > 0:
                not_kelp_slip = jnp.logical_or(self.kelp_map[pos[0], pos[1]] == 0, random.uniform(kelp_key) > self.kelp_prob)
            if self.slip_prob > 0:
                not_normal_slip = random.uniform(normal_key) > self.slip_prob
            not_slips = not_kelp_slip * not_normal_slip
            added_direction = not_slips * self.direction_mapping[action]
        else:
            added_direction = self.direction_mapping[action]

        new_pos = pos + added_direction
        new_pos = jnp.maximum(jnp.minimum(new_pos, self.position_max), self.position_min)

        # if we bump into a wall inside the grid
        in_an_obstacle = self.obstacle_map[new_pos[0], new_pos[1]] > 0
        new_pos = (1 - in_an_obstacle) * new_pos + in_an_obstacle * pos

        return new_pos

    @partial(jax.jit, static_argnums=(0,))
    def tick_currents(self, key: chex.PRNGKey, state: FishingState) -> jnp.ndarray:
        """
        See if we change currents or not. If we do,
        update current map
        """
        current_keys = random.split(key, self.current_probs.shape[0])
        # the 4 here is the number of directions
        new_idxes = self.vmapped_choice(current_keys, self.current_probs.shape[-1], self.current_probs)

        # we - 1 on the new_idxes, since our currents are offset by 1.
        new_current_directions = jax.lax.select(new_idxes == 0,
                                                state.current_directions,
                                                new_idxes - 1)

        return new_current_directions

    @staticmethod
    def opposite_directions(dir_1: int, dir_2: int):
        return jnp.logical_or(jnp.logical_or(jnp.logical_and(dir_1 == 0, dir_2 == 2),
                            jnp.logical_and(dir_1 == 2, dir_2 == 0)),
                            jnp.logical_or(jnp.logical_and(dir_1 == 1, dir_2 == 3),
                            jnp.logical_and(dir_1 == 3, dir_2 == 1)))

    def transition(self, key: chex.PRNGKey, state: FishingState, action: int) -> FishingState:
        # We first move according to our action
        # have it so that you can't move against the current direction if it exists
        # you're currently in (swim orthogonal to the rip!)
        position_current_match = jnp.all(state.position[None, ...] == self.current_positions, axis=-1)

        vmapped_opposite_directions = jax.vmap(self.opposite_directions, in_axes=[None, 0])
        opposing_directions = vmapped_opposite_directions(action, state.current_directions)
        against_the_current = jnp.dot(position_current_match, opposing_directions)

        move_key, key = random.split(key)
        new_pos = jax.lax.select(against_the_current,
                                 state.position,
                                 self.move(move_key, state.position, action, slippage=True))

        # now we move again according to any currents
        no_slip_move = partial(self.move, slippage=False)
        vmapped_move = jax.vmap(no_slip_move, in_axes=[0, None, 0])
        slip_key, key = random.split(key)
        slippage_keys = random.split(slip_key, state.current_directions.shape[0])
        all_current_moved = vmapped_move(slippage_keys, new_pos, state.current_directions)

        new_pos_current_match = jnp.all(new_pos[None, ...] == self.current_positions, axis=-1)

        new_pos = jax.lax.select(jnp.any(new_pos_current_match),
                                 (all_current_moved * new_pos_current_match[..., None]).sum(axis=0),
                                 new_pos)

        current_key, key = random.split(key)
        # now we tick currents
        new_current_directions = self.tick_currents(current_key, state)

        # figure out new active rewards
        reward_keys = random.split(key, self.reward_probs.shape[0])
        new_potential_rewards = self.vmapped_choice(reward_keys, 2, jnp.stack([1 - self.reward_probs, self.reward_probs], axis=-1))
        new_rewards_active = jnp.maximum(new_potential_rewards, state.rewards_active)

        # see if we collect any reward
        position_reward_match = jnp.all(new_pos[None, ...] == self.reward_positions, axis=-1)
        new_rewards_active = jnp.maximum(new_rewards_active - position_reward_match, 0)

        return FishingState(current_directions=new_current_directions,
                            position=new_pos,
                            rewards_active=new_rewards_active,
                            steps=state.steps + 1)

    def step_env(self,
                 key: chex.PRNGKey,
                 state: FishingState,
                 action: int,
                 params: EnvParams):
        transition_key, reward_key = random.split(key)
        next_state = self.transition(transition_key, state, action)

        return lax.stop_gradient(self.get_obs(next_state, params)), \
               lax.stop_gradient(next_state), \
               self.get_reward(reward_key, next_state, state, action),\
               self.get_terminal(next_state, params), {}


