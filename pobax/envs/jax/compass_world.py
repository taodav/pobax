import numpy as np
from jax import random
from gymnasium.spaces import Box, Discrete
from gymnasium import Env
import matplotlib.pyplot as plt

from typing import Tuple

def batch_wall_split(states: np.ndarray, size: int, green_idx: int = 1):
    north_facing_idxes = np.argwhere(states[:, 2] == 0)[:, 0]
    relative_n_facing_wall_idxes = np.argwhere(states[north_facing_idxes][:, 0] == 1)[:, 0]
    north_facing_wall_idxes = north_facing_idxes[relative_n_facing_wall_idxes]

    east_facing_idxes = np.argwhere(states[:, 2] == 1)[:, 0]
    relative_e_facing_wall_idxes = np.argwhere(states[east_facing_idxes][:, 1] == size - 2)[:, 0]
    east_facing_wall_idxes = east_facing_idxes[relative_e_facing_wall_idxes]

    south_facing_idxes = np.argwhere(states[:, 2] == 2)[:, 0]
    relative_s_facing_wall_idxes = np.argwhere(states[south_facing_idxes][:, 0] == size - 2)[:, 0]
    south_facing_wall_idxes = south_facing_idxes[relative_s_facing_wall_idxes]

    west_facing_idxes = np.argwhere(states[:, 2] == 3)[:, 0]
    relative_w_facing_wall_idxes = np.argwhere(states[west_facing_idxes][:, 1] == 1)[:, 0]
    west_facing_wall_idxes = west_facing_idxes[relative_w_facing_wall_idxes]

    relative_w_facing_green_idxes = np.argwhere(states[west_facing_wall_idxes][:, 0] == green_idx)
    relative_w_facing_blue_idxes = np.argwhere(states[west_facing_wall_idxes][:, 0] != green_idx)
    green_facing_wall_idxes = west_facing_wall_idxes[relative_w_facing_green_idxes]
    blue_facing_wall_idxes = west_facing_wall_idxes[relative_w_facing_blue_idxes]

    return north_facing_wall_idxes, east_facing_wall_idxes, south_facing_wall_idxes, blue_facing_wall_idxes, green_facing_wall_idxes


class CompassWorld(Env):
    direction_mapping = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]], dtype=np.int16)

    def __init__(self,
                 rand_key: random.PRNGKey = None,
                 size: int = 8,
                 random_start: bool = True):
        super(CompassWorld, self).__init__()
        self.observation_space = Box(
            low=np.zeros(5), high=np.ones(5)
        )
        self.action_space = Discrete(3)
        self.unique_rewards = [0, 1]
        self.size = size
        self.random_start = random_start

        if rand_key is None:
            rand_key = random.PRNGKey(np.random.randint(1, 100000))
        self.rand_key = rand_key

        self.state_max = [self.size - 2, self.size - 2, 3]
        self.state_min = [1, 1, 0]

        self._state = None

        # This is for particle filter states.
        # We need this def so that we can pass particles and weights between wrappers.
        self.particles = None
        self.weights = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, state: np.ndarray):
        self._state = state

    def batch_transition(self, states: np.ndarray, actions: np.ndarray) -> np.ndarray:
        full_direction_mapping = np.concatenate(
            (self.direction_mapping, np.zeros(self.direction_mapping.shape[0], dtype=np.int16)[:, np.newaxis]), axis=-1)

        directions = np.arange(4)
        next_states = states.copy()

        forward_states = np.argwhere((actions == 0))[:, 0]

        def forward(states: np.ndarray, f_states: np.ndarray, direction: int) -> np.ndarray:
            relative_d_forward_states = np.argwhere(states[f_states][:, 2] == direction)
            d_forward_states = forward_states[relative_d_forward_states][:, 0]
            return d_forward_states

        for d in directions:
            d_forward_states = forward(states, forward_states, d)
            next_states[d_forward_states] += full_direction_mapping[d]

        right_states = np.argwhere((actions == 1))[:, 0]
        next_right_states = next_states[right_states]
        next_right_states[:, 2] = (states[right_states][:, 2] + 1) % 4
        next_states[right_states] = next_right_states

        left_states = np.argwhere((actions == 2))[:, 0]
        next_left_states = next_states[left_states]
        next_left_states[:, 2] = (states[left_states][:, 2] - 1) % 4
        next_states[left_states] = next_left_states

        return np.maximum(np.minimum(next_states, self.state_max), self.state_min)

    def batch_get_obs(self, states: np.ndarray) -> np.ndarray:
        """
        Batch version of get_obs.
        :param state: batch x 3 size
        :return: batch x 5 observation
        """
        assert len(states.shape) == 2
        batch_size = states.shape[0]
        res = np.zeros((batch_size, 5), dtype=np.uint8)

        n_wall, e_wall, s_wall, b_wall, g_wall = batch_wall_split(states, self.size, green_idx=1)

        res[n_wall] = [1, 0, 0, 0, 0]
        res[e_wall] = [0, 1, 0, 0, 0]
        res[s_wall] = [0, 0, 1, 0, 0]
        res[b_wall] = [0, 0, 0, 1, 0]
        res[g_wall] = [0, 0, 0, 0, 1]
        return res

    def get_obs(self, state: np.ndarray) -> np.ndarray:
        obs = np.zeros(5)
        if state[2] == 0:
            # Facing NORTH
            if state[0] == 1:
                obs[0] = 1
        elif state[2] == 1:
            # Facing EAST
            if state[1] == self.size - 2:
                obs[1] = 1
        elif state[2] == 2:
            # Facing SOUTH
            if state[0] == self.size - 2:
                obs[2] = 1
        elif state[2] == 3:
            # Facing WEST
            if state[1] == 1:
                # On the border
                if state[0] == 1:
                    obs[4] = 1
                else:
                    obs[3] = 1
        else:
            raise NotImplementedError()

        return obs

    def get_reward(self) -> int:
        if (self.state == np.array([1, 1, 3])).all():
            return 1
        return 0

    def get_terminal(self) -> bool:
        return (self.state == np.array([1, 1, 3])).all()

    def reset(self) -> np.ndarray:
        """
        State description:
        state[0] -> y-position
        state[1] -> x-position
        state[2] -> direction, with mapping below:
        Mapping: 0 = North, 1 = East, 2 = South, 3 = West

        Observation description:
        Binary vector of size 5, each indicating whether you're directly in front of a certain color or not.
        :return:
        """
        if self.random_start:
            all_states = self.sample_all_states()
            eligible_state_indices = np.arange(0, all_states.shape[0])

            # Make sure to remove goal state from start states
            remove_idx = None
            for i in eligible_state_indices:
                if (all_states[i] == np.array([1, 1, 3])).all():
                    remove_idx = i
            delete_mask = np.ones_like(eligible_state_indices, dtype=bool)
            delete_mask[remove_idx] = False
            eligible_state_indices = eligible_state_indices[delete_mask, ...]
            key, self.rand_key = random.split(self.rand_key)
            start_state_idx = random.choice(key, eligible_state_indices)

            self.state = all_states[start_state_idx]
        else:
            self.state = np.array([3, 3, self.rng.choice(np.arange(0, 4))], dtype=np.int16)

        return self.get_obs(self.state), {'state': self.state.copy()}

    def transition(self, state: np.ndarray, action: int) -> np.ndarray:
        new_state = state.copy()
        if action == 0:
            new_state[:-1] += self.direction_mapping[state[-1]]
        elif action == 1:
            new_state[-1] = (state[-1] + 1) % 4
        elif action == 2:
            new_state[-1] = (state[-1] - 1) % 4

        # Wall interactions
        new_state = np.maximum(np.minimum(new_state, self.state_max), self.state_min)

        return new_state

    def emit_prob(self, states: np.ndarray, obs: np.ndarray) -> np.ndarray:
        """
        Get the probability of emitting a batch of observations given states
        :param state: underlying state, size batch_size x 3
        :param obs: observation, size batch_size x 5
        :return: probability of emitting (either 0 or 1 for this deterministic environment).
        """
        ground_truth_obs = self.batch_get_obs(states)
        return (ground_truth_obs == obs).min(axis=-1).astype(np.float)

    def sample_states(self, n: int = 10) -> np.ndarray:
        """
        Sample n random states (for particle filtering)
        :param n: number of states to sample
        :return: sampled states
        """
        poss = self.rng.choice(np.arange(1, self.size - 1, dtype=np.int16), size=(n, 2))
        dirs = self.rng.choice(np.arange(0, len(self.direction_mapping), dtype=np.int16), size=(n, 1))
        return np.concatenate([poss, dirs], axis=-1)

    def sample_all_states(self) -> np.ndarray:
        """
        Sample all available states
        :return:
        """
        states = np.zeros(((self.size - 2) ** 2 * 4, 3), dtype=np.int16)
        c = 0
        for i in range(1, self.size - 1):
            for j in range(1, self.size - 1):
                for k in range(0, 4):
                    states[c] = np.array([i, j, k])
                    c += 1

        assert c == states.shape[0]

        return states

    def step(self, action: int) -> Tuple[np.ndarray, int, bool, dict]:
        """
        Action description:
        0 = Forward, 1 = Turn Right, 2 = Turn Left
        :param action: Action to take
        :return:
        """
        assert action in list(range(0, 4)), f"Invalid action: {action}"

        self.state = self.transition(self.state, action)

        return self.get_obs(self.state), self.get_reward(), self.get_terminal(), False, {'state': self.state.copy()}

    def generate_array(self) -> np.ndarray:
        """
        Generate a numpy array representing state.
        Mappings for indices are as follows:
        0 = white space
        1 = orange wall
        2 = yellow wall
        3 = red wall
        4 = blue wall
        5 = green wall
        6 = agent facing NORTH
        7 = agent facing EAST
        8 = agent facing SOUTH
        9 = agent facing WEST
        :return:
        """
        viz_array = np.zeros((self.size, self.size), dtype=np.uint8)

        # WEST wall
        viz_array[:, 0] = 4
        viz_array[1, 0] = 5

        # EAST wall
        viz_array[:, self.size - 1] = 2

        # NORTH wall
        viz_array[0, :] = 1

        # SOUTH wall
        viz_array[self.size - 1, :] = 3

        viz_array[self.state[0], self.state[1]] = self.state[-1] + 6
        return viz_array



if __name__ == "__main__":
    env = CompassWorld()
    env.reset()
    plt.imshow(env.render())
    plt.show()