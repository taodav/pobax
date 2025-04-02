from typing import Literal

from jax import numpy as jnp
from tap import Tap


class Hyperparams(Tap):
    env: str = 'tmaze_5'
    alg: Literal['ppo'] = 'ppo'

    default_max_steps_in_episode: int = 1000
    total_steps: int = int(1.5e6)
    gamma: float = 0.99

    num_eval_envs: int = 10
    steps_log_freq: int = 1
    update_log_freq: int = 1
    save_runner_state: bool = False  # Do we save the checkpoint in the end?
    seed: int = 2020
    n_seeds: int = 1
    platform: Literal['cpu', 'gpu'] = 'cpu'
    debug: bool = False

    study_name: str = 'test'


class PPOHyperparams(Tap):
    env: str = 'tmaze_5'
    # env: str = 'Hopper-P-v0'
    # env: str = 'MemoryChain-bsuite'
    num_envs: int = 4
    default_max_steps_in_episode: int = 1000
    gamma: float = 0.99  # will be replaced if env has gamma property.

    num_steps: int = 128
    num_epochs: int = 50
    update_epochs: int = 4
    num_minibatches: int = 4

    memoryless: bool = False
    perfect_memory: bool = False
    double_critic: bool = False
    action_concat: bool = False

    lr: list[float] = [2.5e-4]
    lambda0: list[float] = [0.95]  # GAE lambda_0
    lambda1: list[float] = [0.5]  # GAE lambda_1
    alpha: list[float] = [1.]  # adv = alpha * adv_lambda_0 + (1 - alpha) * adv_lambda_1
    ld_weight: list[float] = [0.0]  # how much to we weight the LD loss vs. value loss? only applies when optimize LD is True.
    vf_coeff: list[float] = [0.5]

    hidden_size: int = 128
    total_steps: int = int(1.5e6)
    entropy_coeff: float = 0.01
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5
    anneal_lr: bool = True

    image_size: int = 32

    num_eval_envs: int = 10
    steps_log_freq: int = 1
    update_log_freq: int = 1
    save_checkpoints: bool = False  # Do we save train_state along with our per timestep outputs?
    save_runner_state: bool = False  # Do we save the checkpoint in the end?
    seed: int = 2020
    n_seeds: int = 5
    platform: Literal['cpu', 'gpu'] = 'cpu'
    debug: bool = False
    show_discounted: bool = False  # For debug plotting, do we show undisc returns or disc returns?

    study_name: str = 'batch_ppo_test'

    def process_args(self) -> None:
        self.vf_coeff = jnp.array(self.vf_coeff)
        self.lr = jnp.array(self.lr)
        self.lambda0 = jnp.array(self.lambda0)
        self.lambda1 = jnp.array(self.lambda1)
        self.alpha = jnp.array(self.alpha)
        self.ld_weight = jnp.array(self.ld_weight)


class DQNHyperparams(Tap):
    env: str = 'CartPole-v1'
    num_envs: int = 4
    gamma: float = 0.99  # will be replaced if env has gamma property.

    buffer_size: int = 10000
    buffer_batch_size: int = 128
    epsilon_start: float = 1.
    epsilon_finish: float = 0.05
    epsilon_anneal_time: int = int(25e4)
    training_interval: int = 10
    target_update_interval: int = 500
    learning_starts: int = 10000
    lr_linear_decay: bool = False

    num_epochs: int = 10

    lr: float = 2.5e-4
    hidden_size: int = 32
    total_steps: int = int(1e6)
    tau: float = 1.

    save_ckpt_per_epoch: bool = False
    seed: int = 2024
    num_seeds: int = 1
    wandb_mode: Literal['disabled', 'online'] = 'disabled'
    debug: bool = False
    platform: Literal['cpu', 'gpu'] = 'cpu'
    study_name: str = 'dqn_test'


class TransformerHyperparams(Tap):
    env: str = 'tmaze_5'
    # env: str = 'Hopper-P-v0'
    # env: str = 'MemoryChain-bsuite'
    num_envs: int = 4
    default_max_steps_in_episode: int = 1000
    gamma: float = 0.99  # will be replaced if env has gamma property.

    num_steps: int = 128
    num_epochs: int = 50
    update_epochs: int = 4
    num_minibatches: int = 4

    double_critic: bool = False
    action_concat: bool = False

    lr: list[float] = [2.5e-4]
    lambda0: list[float] = [0.95]  # GAE lambda_0
    lambda1: list[float] = [0.5]  # GAE lambda_1
    alpha: list[float] = [1.]  # adv = alpha * adv_lambda_0 + (1 - alpha) * adv_lambda_1
    ld_weight: list[float] = [0.0]  # how much to we weight the LD loss vs. value loss? only applies when optimize LD is True.
    vf_coeff: list[float] = [0.5]

    hidden_size: int = 128
    total_steps: int = int(1.5e6)
    entropy_coeff: float = 0.01
    clip_eps: float = 0.2
    max_grad_norm: float = 0.5
    anneal_lr: bool = True

    image_size: int = 32

    num_eval_envs: int = 10
    steps_log_freq: int = 1
    update_log_freq: int = 1
    save_checkpoints: bool = False  # Do we save train_state along with our per timestep outputs?
    save_runner_state: bool = False  # Do we save the checkpoint in the end?
    seed: int = 2020
    n_seeds: int = 5
    platform: Literal['cpu', 'gpu'] = 'cpu'
    debug: bool = False

    # transformer hyperparams
    qkv_features: int = 256
    embed_size: int = 256
    num_heads: int = 8
    num_layers: int = 2
    window_mem: int = 128
    window_grad: int = 64
    gating: bool = True
    gating_bias: float = 2.0

    study_name: str = 'batch_ppo_test'

    def process_args(self) -> None:
        self.vf_coeff = jnp.array(self.vf_coeff)
        self.lr = jnp.array(self.lr)
        self.lambda0 = jnp.array(self.lambda0)
        self.lambda1 = jnp.array(self.lambda1)
        self.alpha = jnp.array(self.alpha)
        self.ld_weight = jnp.array(self.ld_weight)
