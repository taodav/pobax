# POBAX: Partially Observable Benchmarks in JAX
POBAX is a reinforcement learning benchmark that tests all forms of partial observability. 

POBAX has been accepted to RLC 2025. Check out our [paper](https://openreview.net/forum?id=HUTCbYOW5E)!

The benchmark is entirely written in [JAX](https://github.com/jax-ml/jax), allowing for fast, GPU-scalable experimentation.

## Environments
POBAX includes environments (as well as recommended hyperparameter settings) across diverse forms of partial observability. We list our environments from smallest to largest (in terms of neural network size requirements for PPO RNN):

| Environment | Category                                                  | IDs | Description                                                                                                                                                                                                        |
|---|-----------------------------------------------------------|---|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| [Simple Chain](https://github.com/taodav/pobax/blob/main/pobax/envs/jax/simple_chain.py)           | Object uncertainty & tracking                             | `simple_chain` | Diagnostic POMDP for testing algorithms.                                                                                                                                                                           |
| [T-Maze](https://github.com/taodav/pobax/blob/main/pobax/envs/jax/tmaze.py)                        | Object uncertainty & tracking                             | `tmaze_10`                               | [Bakker's](https://papers.nips.cc/paper_files/paper/2001/hash/a38b16173474ba8b1a95bcbc30d3b8a5-Abstract.html) classic memory testing environment (hallway disambiguation).                                         |
| [RockSample](https://github.com/taodav/pobax/blob/main/pobax/envs/jax/rocksample.py)               | Episode nonstationarity and Object uncertainty & tracking | `rocksample_11_11` and `rocksample_15_15` | The classic rock collecting POMDP, where an agent needs to uncover and collect rocks.                                                                                                                              |
| [Battleship](https://github.com/taodav/pobax/blob/main/pobax/envs/jax/battleship.py)               | Object uncertainty & tracking                             | `battleship_10`                          | Single-player battleship (10x10).                                                                                                                                                                                  |
| [Masked Mujoco](https://github.com/taodav/pobax/blob/main/pobax/envs/__init__.py#L98)              | Moment Features                                           | `{env_name}-{F/P/V}-v0`                  | Mujoco with state features masked out. `env_name` can be `Walker`, `Ant`, `Hopper`, or `HalfCheetah`. `F/P/V` stands for fully observable, position only, or velocity only versions of environments, respectively. |
| [DMLab Minigrid](https://github.com/taodav/pobax/blob/main/pobax/envs/jax/navix_mazes.py)          | Spatial uncertainty                                       | `Navix-DMLab-Maze-{01/02/03}-v0`         | [MiniGrid](https://minigrid.farama.org/) versions of the [DeepMind Lab](https://github.com/google-deepmind/lab) mazes. `01/02/03` refer to the DeepMind Lab Minigrid mazes in ascending difficulty.                |
| [Visual Continuous Control]() | Visual occlusion and moment features                      | `{env_name}-pixels`                      | Pixel-based versions of Mujoco control. **Requires the [Madrona_MJX](https://github.com/shacklettbp/madrona_mjx) package**. `env_name` can be `ant`, `halfcheetah`, `hopper`, or `walker2d`.                       |
| [No-Inventory Crafter](https://github.com/taodav/pobax/blob/main/pobax/envs/__init__.py#L112)    | Object uncertainty & tracking and spatial uncertainty     | `craftax-pixels`                         | Crafter without the inventory. **Requires the [Craftax](https://github.com/MichaelTMatthews/Craftax) package**.                                                                                                    |

<br>

Experimental results on memory-based deep reinforcement learning algorithms are shown here and in [our work](https://openreview.net/forum?id=HUTCbYOW5E).

![POBAX experimental results](https://github.com/taodav/pobax/blob/main/images/all_envs.png?raw=true)

## Basic Usage
```python
import jax
from pobax.envs import get_env

rand_key = jax.random.PRNGKey(2025)
env_key, rand_key = jax.random.split(rand_key)

# Creates a vectorized environment
env, env_params = get_env("rocksample_11_11", env_key)

# Reset 10 environments
reset_key, rand_key = jax.random.split(rand_key)
reset_keys = jax.random.split(rand_key, 10)

obs, env_state = env.reset(reset_keys, env_params)

# Take steps in all environments
step_key, action_key, rand_key = jax.random.split(rand_key, 3)
step_keys = jax.random.split(step_key, 10)
action_keys = jax.random.split(action_key, 10)

actions = jax.vmap(env.action_space(env_params).sample)(action_keys)

obs, env_state, reward, done, info = env.step(step_keys, env_state, actions, env_params)
```

## Installation

The latest `pobax` version can be installed via PyPI:
```shell
pip install pobax
```
To develop for the `pobax` package, create a fork and clone the repo:
```shell
git clone git@github.com:{FORKED_USER}/pobax.git
cd pobax
pip install -e .
```

### Installing Madrona_MJX (Optional)

POBAX's pixel-based continuous control environments (`ant-pixels`, `halfcheetah-pixels`, `hopper-pixels`, `walker2d-pixels`) require the [Madrona_MJX](https://github.com/shacklettbp/madrona_mjx) renderer for GPU-accelerated rendering.

**Installation:**
```shell
git clone https://github.com/KevinGuo27/madrona_mjx.git
cd madrona_mjx
pip install -e .
```

**Requirements:**
- CUDA 12.6.3 or compatible version
- GPU support required

**Note:** Madrona_MJX currently does not support `jax.vmap`, so experiments must run with a single seed at a time. See `scripts/hyperparams/visual_mujoco/ant/best/ant_ppo_madrona_best.py` for an example configuration.

**Compilation:** The first time you run a Madrona_MJX environment, the renderer will compile (takes ~4 minutes on an RTX 3090). You'll see output like this:
```
Using raytracer
Compiling .../madrona/src/mw/device/bvh.cpp
Compiling .../madrona/src/mw/device/memory.cpp
Compiling .../madrona/src/mw/device/host_print.cpp
Compiling .../madrona/src/mw/device/bvh_raycast.cpp
Compiling GPU engine code:
Initialization finished
```
Here is an example script to run a Madrona_MJX environment:

## Agents

POBAX includes algorithms loosely based on the [PureJAXRL](https://github.com/luchris429/purejaxrl/tree/main/purejaxrl) framework, with algorithms based on [proximal policy optimization (PPO)](https://arxiv.org/abs/1707.06347). These include:
* Recurrent PPO,
* [λ-discrepancy](https://arxiv.org/abs/2407.07333),
* [GTrXL](https://arxiv.org/abs/1910.06764).

Memoryless versions of the recurrent PPO algorithm is also included with the `--memoryless` flag.

Here's an example script of how to run a recurrent PPO agent on T-Maze:
```shell
python -m pobax.algos.ppo --env tmaze_5 --debug
```

Here's a small example of how to sweep hyperparameters using recurrent PPO agent in RockSample(11, 11):


```shell

python -m pobax.algos.ppo --env rocksample_11_11 --num_envs 16 --entropy_coeff 0.2 --lambda0 0.7 --lr 0.0025 0.00025 --total_steps 5000000 --seed 2024 --platform cpu --n_seeds 5 --debug

```

This script will run an experiment over 5 seeds over 5M steps on CPU with `entropy coefficient = 0.2`, `GAE lambda = 0.7` and 16 parallel environments for each run,
while sweeping `learning rate = 0.0025, 0.00025`.

Here's an example of how to run a pixel-based ant environment (requires Madrona_MJX):
```shell
python -m pobax.algos.ppo --env ant_pixels --action_concat --lambda0 0.7 --lambda1 0.95 --hidden_size 512 --total_steps 5000000 --n_seeds 1 --platform gpu --debug --study_name ant_ppo_madrona_best
```

Hyperparameters and their descriptions can be found in `pobax/config.py`. Any hyperparameter that has a list type can be swept.

## Citation
```
@article{tao2025pobax,
  author = {Tao, Ruo Yu and Guo, Kaicheng and Allen, Cameron and Konidaris, George},
  title = {Benchmarking Partial Observability in Reinforcement Learning with a Suite of Memory-Improvable Domains},
  booktitle = {Proceedings of the Second Reinforcement Learning Conference},
  journal = {The Reinforcement Learning Journal}
  url = {http://github.com/taodav/pobax},
  year = {2025},
}
```
