from pathlib import Path

import gymnax
from jax import random

from definitions import ROOT_DIR

from .battleship import Battleship
from .classic import load_pomdp, load_pomdp
from .compass_world import CompassWorld
from .fishing import Fishing
from .pocman import PocMan
from .rocksample import RockSample
from .reacher_pomdp import ReacherPOMDP
from .simple_chain import SimpleChain, FullyObservableSimpleChain
from .tmaze import TMaze
from .wrappers import (
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
    StackObservationWrapper,
    ConcatRecentObservationsWrapper
)


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

brax_envs = ['ant', 'walker2d', 'halfcheetah', 'hopper']

def load_brax_env(env_str: str,
                  gamma: float = 0.99):
    from gymnax import EnvParams
    from .wrappers import BraxGymnaxWrapper, LogWrapper, ClipAction, VecEnv
    from .wrappers import NormalizeVecReward, NormalizeVecObservation
    env = BraxGymnaxWrapper(env_str)
    env_params = EnvParams(max_steps_in_episode=env.max_steps_in_episode)
    env = ClipAction(env)
    return env, env_params


def get_env(env_name: str,
                   rand_key: random.PRNGKey,
                   normalize_env: bool = False,
                   gamma: float = 0.99,
                   action_concat: bool = False,
                   num_stacks: int = 1,
                   num_observations: int = 1):

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

    elif env_name == 'pocman':
        env = PocMan()
        env_params = env.default_params

    elif env_name == 'ReacherPOMDP':
        env = ReacherPOMDP()
        env_params = env.default_params

    elif 'fishing' in env_name:
        config_path = envs_dir / 'configs' / 'ocean_nav' / f'{env_name}_config.json'
        env = Fishing(config_path=config_path)
        env_params = env.default_params

    elif env_name in brax_envs:
        env, env_params = load_brax_env(env_name, gamma=gamma)

    elif 'rocksample' in env_name:  # [rocksample, rocksample_15_15]

        if len(env_name.split('_')) > 1:
            config_path = Path(ROOT_DIR, 'porl', 'envs', 'configs', f'{env_name}_config.json')
            env = RockSample(rand_key, config_path=config_path)
        else:
            env = RockSample(rand_key)
        env_params = env.default_params

    else:
        env, env_params = gymnax.make(env_name)
        env = FlattenObservationWrapper(env)

    if hasattr(env, 'gamma'):
        print(f"Overwriting args gamma {gamma} with env gamma {env.gamma}.")
        gamma = env.gamma

    if action_concat:
        env = ActionConcatWrapper(env)

    env = LogWrapper(env, gamma=gamma)

    if mask_dims is not None:
        env = MaskObservationWrapper(env, mask_dims=mask_dims)

    if num_stacks > 1:
        env = StackObservationWrapper(env, num_stack=num_stacks)
        env = FlattenObservationWrapper(env)
    
    if num_observations > 1:
        env = ConcatRecentObservationsWrapper(env, num_recent_observations=num_observations)

    # Vectorize our environment
    env = VecEnv(env)
    if normalize_env:
        env = NormalizeVecObservation(env)
        env = NormalizeVecReward(env, gamma)
    elif 'rocksample' in env_name:
        env = NormalizeVecReward(env, gamma)
    
    return env, env_params



