from typing import Any

from gymnasium import Wrapper
from gymnasium.core import WrapperObsType, WrapperActType, SupportsFloat

from gymnasium.wrappers import PixelObservationWrapper


class PixelOnlyObservationWrapper(PixelObservationWrapper):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.observation_space = self.observation_space['pixels']

    def observation(self, observation):
        dict_observations = super().observation(observation)
        return dict_observations['pixels']


class OnlineReturnsLogWrapper(Wrapper):
    def __init__(self, *args, gamma: float = 0.9, **kwargs):
        super().__init__(*args, **kwargs)
        self.gamma = gamma
        self.episode_return = 0
        self.discounted_episode_return = 0
        self.episode_length = 0
        self.returned_episode_return = 0
        self.returned_discounted_episode_return = 0
        self.timestep = 0

    def reset(
        self, *, seed: int | None = None, options: dict[str, Any] | None = None
    ) -> tuple[WrapperObsType, dict[str, Any]]:
        obs, info = self.env.reset(seed=seed, options=options)
        self.returned_episode_return = self.episode_return
        self.returned_discounted_episode_return = self.discounted_episode_return
        info = {
            'returned_episode_return': self.returned_episode_return,
            'returned_discounted_episode_return': self.returned_discounted_episode_return,
            'episode_length': self.episode_length,
            **info
        }

        self.episode_return = 0
        self.discounted_episode_return = 0
        self.episode_length = 0
        self.timestep = 0
        return obs, info

    def step(
        self, action: WrapperActType
    ) -> tuple[WrapperObsType, SupportsFloat, bool, bool, dict[str, Any]]:
        obs, reward, done, truncation, info = self.env.step(action)

        new_episode_return = self.episode_return + reward
        new_discounted_episode_return = self.discounted_episode_return + (self.gamma ** self.episode_length) * reward
        new_episode_length = self.episode_length + 1

        not_done_or_not_trunc = (1 - done) * (1 - truncation)
        self.episode_return = new_episode_return * not_done_or_not_trunc
        self.discounted_episode_return = new_discounted_episode_return * not_done_or_not_trunc
        self.episode_length = new_episode_length * not_done_or_not_trunc

        self.returned_episode_return = self.returned_episode_return * not_done_or_not_trunc \
                                       + new_episode_return * (1 - not_done_or_not_trunc)
        self.returned_discounted_episode_return = self.returned_discounted_episode_return * not_done_or_not_trunc\
                                                  + new_discounted_episode_return * (1 - not_done_or_not_trunc)

        info = {
            'episode_return': self.episode_return,
            'returned_episode_return': self.returned_episode_return,
            'returned_discounted_episode_return': self.returned_discounted_episode_return,
            'episode_length': self.episode_length,
            # **info
        }
        return obs, reward, done, truncation, info

