from pathlib import Path
from time import time
from typing import Union
from tqdm import trange
import numpy as np

from pobax.envs.gymnasium.rocksample import RockSample, GlobalStateObservationWrapper
from pobax.envs.gymnasium.battleship import BattleshipEnvClass

from tap import Tap
from definitions import ROOT_DIR


class SampleHyperparams(Tap):
    env: str = 'rocksample'
    n_envs: int = 1
    n_steps: Union[int, str] = int(1e7)

    seed: int = 2024
    platform: str = 'cpu'

    def configure(self) -> None:
        def to_int(s):
            return int(float(s))
        self.add_argument('--n_steps', type=to_int)


if __name__ == "__main__":
    args = SampleHyperparams().parse_args()

    env = None
    if args.env == 'rocksample':
        config_path = Path(ROOT_DIR, 'pobax/envs/configs/rocksample_11_11_config.json')
        env = RockSample(config_path, np.random.RandomState(args.seed))
        env = GlobalStateObservationWrapper(env)
    elif args.env == 'battleship':
        env = BattleshipEnvClass()

    obs = env.reset()
    done = False
    t = time()
    step = 0
    for i in trange(args.n_steps):
        next_obs, rew, done, info = env.step(env.action_space.sample())
        step += 1
        if done:
            obs = env.reset()

    new_t = time()
    total_runtime = new_t - t
    print(f'Total runtime for {args.env} environment with {args.n_envs} envs and {args.n_steps} steps:', total_runtime)
