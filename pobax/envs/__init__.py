from functools import partial
from pathlib import Path

import gymnax
from jax import random

from definitions import ROOT_DIR

from pobax.envs.jax.battleship import Battleship
from pobax.envs.jax.battleship import PerfectMemoryWrapper as BSPerfectMemoryWrapper
from pobax.envs.classic import load_pomdp, load_pomdp
from pobax.envs.jax.compass_world import CompassWorld
from pobax.envs.jax.fishing import Fishing
from pobax.envs.jax.pocman import PocMan
from pobax.envs.jax.pocman import PerfectMemoryWrapper as PMPerfectMemoryWrapper
from pobax.envs.jax.rocksample import RockSample
from pobax.envs.jax.rocksample import PerfectMemoryWrapper as RSPerfectMemoryWrapper
from pobax.envs.jax.reacher_pomdp import ReacherPOMDP
from pobax.envs.jax.simple_chain import SimpleChain
from pobax.envs.jax.tmaze import TMaze
from pobax.envs.wrappers.gymnax import (
    FlattenObservationWrapper,
    LogWrapper,
    MaskObservationWrapper,
    BraxGymnaxWrapper,
    LogWrapper,
    ClipAction,
    VecEnv,
    NormalizeVecReward,
    NormalizeVecObservation,
    ActionConcatWrapper,
    AutoResetEnvWrapper
)
from pobax.envs.wrappers.pixel import PixelBraxVecEnvWrapper, PixelTMazeVecEnvWrapper, PixelSimpleChainVecEnvWrapper, PixelCraftaxVecEnvWrapper
from pobax.envs.wrappers.gymnasium import GymnaxToGymWrapper

masked_gymnax_env_map = {
    'Pendulum-F-v0': {'env_str': 'Pendulum-v1', 'mask_dims': [0, 1, 2]},
    'Pendulum-P-v0': {'env_str': 'Pendulum-v1', 'mask_dims': [0, 1]},
    'Pendulum-V-v0': {'env_str': 'Pendulum-v1', 'mask_dims': [2]},
    'CartPole-F-v0': {'env_str': 'CartPole-v1', 'mask_dims': [0, 1, 2, 3]},
    'CartPole-P-v0': {'env_str': 'CartPole-v1', 'mask_dims': [0, 2]},
    'CartPole-V-v0': {'env_str': 'CartPole-v1', 'mask_dims': [1, 3]},
    'LunarLander-F-v0': {'env_str': 'LunarLander-v2', 'mask_dims': list(range(8))},
    'LunarLander-P-v0': {'env_str': 'LunarLander-v2', 'mask_dims': [0, 1, 4, 6, 7]},
    'LunarLander-V-v0': {'env_str': 'LunarLander-v2', 'mask_dims': [2, 3, 5, 6, 7]},
    "Hopper-F-v0": {'env_str': 'hopper', 'mask_dims': list(range(11))},
    "Hopper-P-v0": {'env_str': 'hopper', 'mask_dims': [0, 1, 2, 3, 4]},
    "Hopper-V-v0": {'env_str': 'hopper', 'mask_dims': [5, 6, 7, 8, 9, 10]},
    "Walker-F-v0": {'env_str': 'walker2d', 'mask_dims': list(range(17))},
    "Walker-P-v0": {'env_str': 'walker2d', 'mask_dims': [0, 1, 2, 3, 4, 5, 6, 7]},
    "Walker-V-v0": {'env_str': 'walker2d', 'mask_dims': [8, 9, 10, 11, 12, 13, 14, 15, 16]},
    "Ant-F-v0": {'env_str': 'ant', 'mask_dims': list(range(27))},
    "Ant-P-v0": {'env_str': 'ant', 'mask_dims': [
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]},
    "Ant-V-v0": {'env_str': 'ant', 'mask_dims': [13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]},
    "HalfCheetah-F-v0": {'env_str': 'halfcheetah', 'mask_dims': list(range(17))},
    "HalfCheetah-P-v0": {'env_str': 'halfcheetah', 'mask_dims': [
        0, 1, 2, 3, 8, 9, 10, 11, 12]},
    "HalfCheetah-V-v0": {'env_str': 'halfcheetah', 'mask_dims': [4, 5, 6, 7, 13, 14, 15, 16]},
}


brax_envs = ['ant', 'walker2d', 'halfcheetah', 'hopper', 'ant_pixels', 'walker2d_pixels', 'halfcheetah_pixels', 'hopper_pixels']

# craftax_envs = ['Craftax-Symbolic-v1', 'Craftax-Pixels-v1', 'Craftax-Classic-Symbolic-v1', 'Craftax-Classic-Pixels-v1']
craftax_envs = {'craftax': 'Craftax-Symbolic-v1', 'craftax_pixels': 'Craftax-Pixels-v1', 'craftax_classic': 'Craftax-Classic-Symbolic-v1', 'craftax_classic_pixels': 'Craftax-Classic-Pixels-v1'}

def is_jax_env(env_name: str):
    from brax.envs import _envs as brax_envs
    is_masked_gymnax_env = env_name in masked_gymnax_env_map
    is_brax_env = env_name in brax_envs

    envs_dir = Path(ROOT_DIR) / 'pobax' / 'envs'
    pomdp_dir = envs_dir / 'classic' / 'POMDP'
    pomdp_files = [pd.stem for pd in pomdp_dir.iterdir()]
    is_pomdp_env = env_name in pomdp_files

    is_implemented_env = env_name.startswith('battleship') or env_name == 'pocman' or env_name == 'ReacherPOMDP' \
                            or 'fishing' in env_name or 'rocksample' in env_name

    is_gymnax_env = True
    try:
        gymnax.make(env_name)
    except ValueError as e:
        is_gymnax_env = False

    all_bools = [is_masked_gymnax_env, is_gymnax_env, is_pomdp_env, is_implemented_env, is_brax_env]

    return any(all_bools)


def load_brax_env(env_str: str,
                  gamma: float = 0.99):
    from gymnax import EnvParams
    from pobax.envs.wrappers.gymnax import BraxGymnaxWrapper, LogWrapper, ClipAction, VecEnv
    from pobax.envs.wrappers.gymnax import NormalizeVecReward, NormalizeVecObservation
    if env_str.endswith('pixels'):
        env_str = env_str.split('_')[0]
    env = BraxGymnaxWrapper(env_str)
    env_params = EnvParams(max_steps_in_episode=env.max_steps_in_episode)

    env = ClipAction(env)

    return env, env_params

def load_craftax_env(env_str: str,
                        gamma: float = 0.99):
    from gymnax import EnvParams
    from pobax.envs.wrappers.gymnax import CraftaxGymnaxWrapper
    env_str = craftax_envs[env_str]
    env = CraftaxGymnaxWrapper(env_str)
    env = AutoResetEnvWrapper(env)
    env_params = env.env_params
    return env, env_params


def get_env(env_name: str,
            rand_key: random.PRNGKey,
            normalize_env: bool = False,
            normalize_image: bool = True,
            gamma: float = 0.99,
            perfect_memory: bool = False,
            action_concat: bool = False):

    mask_dims = None
    if env_name in masked_gymnax_env_map:
        spec = masked_gymnax_env_map[env_name]
        env_name = spec['env_str']
        mask_dims = spec['mask_dims']
    envs_dir = Path(ROOT_DIR) / 'pobax' / 'envs'

    pomdp_dir = envs_dir / 'classic' / 'POMDP'
    pomdp_files = [pd.stem for pd in pomdp_dir.iterdir()]

    fo_pomdp = 'fully_observable' in env_name
    if fo_pomdp:
        env_name = env_name.split('_')[-1]

    if env_name.startswith('tmaze_'):
        hallway_length = int(env_name.split('_')[-1])
        env = TMaze(hallway_length=hallway_length)
        env_params = env.default_params

    elif env_name in pomdp_files:
        env = load_pomdp(env_name, fully_observable=fo_pomdp)
        if hasattr(env, 'gamma'):
            gamma = env.gamma
        env_params = env.default_params
    elif env_name.startswith('battleship'):
        rows = cols = 10
        ship_lengths = (5, 4, 3, 2)
        if env_name == 'battleship_5':
            rows = cols = 5
            ship_lengths = (3, 2)
        elif env_name == 'battleship_3':
            rows = cols = 3
            ship_lengths = (2, )

        env = Battleship(rows=rows, cols=cols, ship_lengths=ship_lengths)
        env_params = env.default_params

        if perfect_memory:
            env = BSPerfectMemoryWrapper(env)

    elif env_name == 'pocman':
        env = PocMan()
        env_params = env.default_params

        if perfect_memory:
            env = PMPerfectMemoryWrapper(env)

    elif env_name == 'ReacherPOMDP':
        env = ReacherPOMDP()
        env_params = env.default_params

    elif 'fishing' in env_name:
        config_path = envs_dir / 'configs' / 'ocean_nav' / f'{env_name}_config.json'
        env = Fishing(config_path=config_path)
        env_params = env.default_params

    elif env_name in brax_envs:
        env, env_params = load_brax_env(env_name, gamma=gamma)
    
    elif env_name in craftax_envs:
        env, env_params = load_craftax_env(env_name, gamma=gamma)

    elif 'rocksample' in env_name:  # [rocksample, rocksample_15_15]

        if len(env_name.split('_')) > 1:
            config_path = Path(ROOT_DIR, 'pobax', 'envs', 'configs', f'{env_name}_config.json')
            env = RockSample(rand_key, config_path=config_path)
        else:
            env = RockSample(rand_key)
        env_params = env.default_params
        if perfect_memory:
            env = RSPerfectMemoryWrapper(env)

    else:
        env, env_params = gymnax.make(env_name)
        env = FlattenObservationWrapper(env)

    if hasattr(env, 'gamma'):
        print(f"Overwriting args gamma {gamma} with env gamma {env.gamma}.")
        gamma = env.gamma

    if action_concat and len(env.observation_space(env_params).shape) == 1:
        env = ActionConcatWrapper(env)

    env = LogWrapper(env, gamma=gamma)

    if mask_dims is not None:
        env = MaskObservationWrapper(env, mask_dims=mask_dims)

    # Vectorize our environment
    env = VecEnv(env)
    if env_name.endswith('pixels'):
        env = PixelCraftaxVecEnvWrapper(env, normalize=normalize_image)

    if normalize_env:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, gamma)
    elif 'rocksample' in env_name:
        env = NormalizeVecReward(env, gamma)
    return env, env_params


def get_gym_env(env_name: str,
                gamma: float = 0.99,
                image_size: int = 64,
                normalize_image: bool = True,
                num_envs: int = None,
                seed: int = 2024):
    # For testing purposes
    if env_name == 'tmaze':
        # hallway_length = int(env_name.split('_')[-1])
        env = TMaze(hallway_length=5)
        env_params = env.default_params

        env = LogWrapper(env, gamma=gamma)
        env = VecEnv(env)
        env = PixelTMazeVecEnvWrapper(env, size=image_size, normalize=normalize_image)
    elif env_name == 'simple_chain':
        env = SimpleChain()
        env_params = env.default_params
        env = LogWrapper(env, gamma=gamma)
        env = VecEnv(env)
        env = PixelSimpleChainVecEnvWrapper(env, size=image_size, normalize=normalize_image)
    elif env_name in brax_envs:
        env, env_params = load_brax_env(env_name)
        env = LogWrapper(env, gamma=gamma)
        env = VecEnv(env)
        if env_name.endswith('pixels'):
            env = PixelBraxVecEnvWrapper(env, size=image_size, normalize=normalize_image)
    else:
        env, env_params = load_craftax_env(env_name)
        env = LogWrapper(env, gamma=gamma)
        env = VecEnv(env)
        if env_name.endswith('pixels'):
            env = PixelCraftaxVecEnvWrapper(env, normalize=normalize_image)
    env = GymnaxToGymWrapper(env, env_params, seed=seed, num_envs=num_envs)
    return env
