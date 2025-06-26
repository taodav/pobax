# POBAX: Partially Observable Benchmarks in JAX
***
POBAX is a reinforcement learning benchmark that tests all forms of partial observability. 

POBAX has been accepted to RLC 2025. Check out our [paper](https://openreview.net/forum?id=HUTCbYOW5E)!

The benchmark is entirely written in [JAX](https://github.com/jax-ml/jax), allowing for fast, GPU-scalable experimentation.

## Environments
***
POBAX includes environments (as well as recommended hyperparameter settings) across diverse forms of partial observability. We list our environments from smallest to largest (in terms of neural network size requirements for PPO RNN):

| Environment                                                                                          | Category | IDs                                       | Source | Description                                                                                                                                                                 |
|------------------------------------------------------------------------------------------------------|---|-------------------------------------------|--------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Simple Chain](https://github.com/taodav/pobax/blob/main/pobax/envs/jax/simple_chain.py)             |      | `simple_chain`                            |        | Diagnostic POMDP.                                                                                                                                                           |
| [T-Maze](https://github.com/taodav/pobax/blob/main/pobax/envs/jax/tmaze.py)                          |      | `tmaze_10`                                |        |                                                                                                                                                                             |
| [RockSample](https://github.com/taodav/pobax/blob/main/pobax/envs/jax/rocksample.py)                 |      | `rocksample_11_11` and `rocksample_15_15` |        |                                                                                                                                                                             |
| [Battleship](https://github.com/taodav/pobax/blob/main/pobax/envs/jax/battleship.py)                 |      | `battleship_10`                           |        |                                                                                                                                                                             |
| [Masked Mujoco](https://github.com/taodav/pobax/blob/main/pobax/envs/__init__.py#L98)                |      | `{env_name}-{F/P/V}-v0`                   |        | `env_name` can be `Walker`, `Ant`, `Hopper`, or `HalfCheetah`. `F/P/V` stands for fully observable, position only, or velocity only versions of environments, respectively. |
| [DMLab Minigrid](https://github.com/taodav/pobax/blob/main/pobax/envs/jax/navix_mazes.py)            |      | `Navix-DMLab-Maze-{01/02/03}-v0`          |        | `01/02/03` refer to the DMLab Minigrid mazes in ascending difficulty.                                                                                                       |
| [Visual Continuous Control]() |      | `{env_name}-pixels`                       |        | **Requires the [Madrona_MJX](https://github.com/shacklettbp/madrona_mjx) package**. `env_name` can be `ant`, `halfcheetah`, `hopper`, or `walker2d`.                        |
| [No-Inventory Crafter](https://github.com/taodav/pobax/blob/main/pobax/envs/__init__.py#L112)      |      | `craftax-pixels`                          |        | **Requires the [Craftax](https://github.com/MichaelTMatthews/Craftax) package**.                                                                                             |
