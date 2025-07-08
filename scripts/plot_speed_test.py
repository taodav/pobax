import matplotlib.pyplot as plt
from matplotlib import rc

from pobax.utils.plot import colors

rc('font', **{'family': 'serif', 'serif': ['cmr10'], 'size': 32})
rc('axes', unicode_minus=False)

env_to_name = {
    'rocksample_11_11': 'RockSample(11, 11)',
    'battleship_10': 'BattleShip',
    'Walker-V-v0': 'Walker (Vel. Only)'
}

data = [
  {
    "env": "rocksample_11_11",
    "n_envs": [1, 5, 25, 125],
    "gpu_timings": [393.9228, 405.1604, 406.4938, 497.4396],
    "cpu_timing_single_env": 393.3330,
    "gym_timing_single_env": 467.5841
  },
  {
    "env": "battleship_10",
    "n_envs": [1, 5, 25, 125],
    "gpu_timings": [704.5981, 764.3613, 846.0283, 846.7103],
    "cpu_timing_single_env": 703.5469,
    "gym_timing_single_env": 103.7113
  },
  {
    "env": "Walker-V-v0",
    "n_envs": [1, 5, 25, 125],
    "gpu_timings": [8553.0134, 9003.7577, 9701.4790, 10267.092],
    "cpu_timing_single_env": 8695.7786,
    "gym_timing_single_env": 4836.4002
  }
]


if __name__ == "__main__":

    # Create a row of subplots
    fig, axes = plt.subplots(1, len(data), figsize=(8 * len(data), 7), sharey=True)

    for ax, env_data in zip(axes, data):
        env = env_data["env"]
        n_envs = env_data["n_envs"]
        gpu = env_data["gpu_timings"]
        cpu = env_data["cpu_timing_single_env"]
        gym = env_data["gym_timing_single_env"]

        # Plot GPU timings vs number of envs
        ax.plot(gpu, n_envs, marker='o', linestyle='-', color=colors['blue'], label='gpu')

        # Vertical lines for CPU and Gym single-env timings
        ax.axvline(cpu, color=colors['purple'], linestyle='--', label='cpu')
        ax.axvline(gym, color=colors['orange'], linestyle='-.', label='gym')

        # Labels and title
        ax.set_title(env_to_name[env])
        ax.set_xlabel('Time for 10^7 steps (seconds)')
        ax.grid(True, which='both', axis='both', linestyle=':', linewidth=0.5)
        # ax.legend()

    # Common y-label
    axes[0].set_ylabel('Number of environments')

    plt.tight_layout()
    plt.show()
