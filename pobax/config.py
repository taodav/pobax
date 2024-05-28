from typing import Literal

from jax import numpy as jnp
from tap import Tap


class BatchHyperparams(Tap):
    env: str = 'CartPole-P-v0'

    default_max_steps_in_episode: int = 1000
    memoryless: bool = False
    action_concat: bool = False

    lr: list[float] = [2.5e-4]
    lambda0: list[float] = [0.95]  # GAE lambda_0
    vf_coeff: list[float] = [0.5]
    entropy_coeff: list[float] = [0.01]

    hidden_size: int = 128
    total_steps: int = int(1.5e6)


    steps_log_freq: int = 1
    update_log_freq: int = 1
    save_runner_state: bool = False  # Do we save the checkpoint in the end?
    seed: int = 2020
    n_seeds: int = 5
    platform: Literal['cpu', 'gpu'] = 'cpu'
    debug: bool = False

    study_name: str = 'test'

    def process_args(self) -> None:
        self.vf_coeff = jnp.array(self.vf_coeff)
        self.lr = jnp.array(self.lr)
        self.lambda0 = jnp.array(self.lambda0)
