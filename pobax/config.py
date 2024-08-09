from typing import Literal

from jax import numpy as jnp
from tap import Tap


class Hyperparams(Tap):
    env: str = 'CartPole-v1'
    alg: Literal['ppo'] = 'ppo'

    default_max_steps_in_episode: int = 1000
    total_steps: int = int(1e7)
    gamma: float = 0.99

    num_eval_envs: int = 10
    steps_log_freq: int = 1
    update_log_freq: int = 1
    save_runner_state: bool = False  # Do we save the checkpoint in the end?
    seed: int = 2021
    n_seeds: int = 1
    platform: Literal['cpu', 'gpu'] = 'cpu'
    debug: bool = False

    study_name: str = 'test'


class PPOHyperparams(Hyperparams):
    perfect_memory: bool = False
    memoryless: bool = False
    action_concat: bool = False
    double_critic: bool = False
    
    # fix rnn params
    approximator: Literal['mlp', 'rnn'] = 'mlp'
    horizon: int = 3
    num_stack: int = 1
    num_observation: int = 1

    lr: list[float] = [2.5e-4] # learning rate
    lambda0: list[float] = [0.95]  # GAE lambda_0
    lambda1: list[float] = [0.5]  # GAE lambda_1
    alpha: list[float] = [1.]  # adv = alpha * adv_lambda_0 + (1 - alpha) * adv_lambda_1
    ld_weight: list[float] = [0.0]  # how much to we weight the LD loss vs. value loss? only applies when optimize LD is True.
    vf_coeff: list[float] = [0.5]

    entropy_coeff: float = 0.01
    clip_eps: float = 0.2  # PPO log grad clipping
    max_grad_norm: float = 0.5

    not_anneal_lr: bool = True
    hidden_size: int = 128
    depth: int = 3
    num_minibatches: int = 4
    num_envs: int = 4
    num_steps: int = 128
    update_epochs: int = 4

    def process_args(self) -> None:
        self.vf_coeff = jnp.array(self.vf_coeff)
        self.lr = jnp.array(self.lr)
        self.lambda0 = jnp.array(self.lambda0)
        self.lambda1 = jnp.array(self.lambda1)
        self.alpha = jnp.array(self.alpha)
        self.ld_weight = jnp.array(self.ld_weight)
        self.entropy_coeff = jnp.array(self.entropy_coeff)
        self.clip_eps = jnp.array(self.clip_eps)
        self.max_grad_norm = jnp.array(self.max_grad_norm)

        self.num_updates = self.total_steps // self.num_steps // self.num_envs
        self.minibatch_size = self.num_envs * self.num_steps // self.num_minibatches

