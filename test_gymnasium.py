from functools import partial

import jax

import gymnasium as gym
from gymnasium.wrappers import NormalizeObservation, ResizeObservation

from pobax.envs.wrappers.gymnasium import PixelOnlyObservationWrapper, OnlineReturnsLogWrapper

def init_env(env_name: str):
    # try out pixel observations for brax
    env = gym.make(env_name, render_mode='rgb_array')
    env = gym.wrappers.AutoResetWrapper(env)
    env = PixelOnlyObservationWrapper(env)
    env = ResizeObservation(env, shape=64)
    env = NormalizeObservation(env)
    env = OnlineReturnsLogWrapper(env)
    return env

def init_vector_env(env_name: str, num_envs: int = 4):
    wrappers = [
        # gym.wrappers.AutoResetWrapper,
        PixelOnlyObservationWrapper,
        partial(ResizeObservation, shape=64),
        NormalizeObservation,
        OnlineReturnsLogWrapper
    ]

    env = gym.vector.make(env_name, num_envs=num_envs, wrappers=wrappers, render_mode='rgb_array')
    return env


if __name__ == "__main__":

    env_name = 'HalfCheetah-v4'

    # env = init_env(env_name)
    env = init_vector_env(env_name)

    jax.config.update('jax_platform_name', 'gpu')

    a = jax.numpy.zeros(3)
    obs, info = env.reset()

    infos = []
    for i in range(2000):
        obs, reward, done, truncation, info = env.step(env.action_space.sample())
        infos.append(info)

    print()
