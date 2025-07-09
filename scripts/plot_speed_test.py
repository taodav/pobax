from pathlib import Path
import matplotlib.pyplot as plt
from matplotlib import rc

from pobax.utils.plot import colors
from definitions import ROOT_DIR

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
    "n_envs": [1, 5, 25, 125, 2000],
    "gpu_timings": [393.9228, 405.1604, 406.4938, 497.4396, 500.7241],
    "cpu_timing_single_env": 393.3330,
    "gym_timing_single_env": 467.5841
  },
  {
    "env": "battleship_10",
    "n_envs": [1, 5, 25, 125, 2000],
    "gpu_timings": [704.5981, 764.3613, 846.0283, 846.7103, 1116.8773],
    "cpu_timing_single_env": 703.5469,
    "gym_timing_single_env": 103.7113
  },
  {
    "env": "Walker-V-v0",
    "n_envs": [1, 5, 25, 125, 2000],
    "gpu_timings": [8553.0134, 9003.7577, 9701.4790, 10267.092, 15228.2229],
    "cpu_timing_single_env": 8695.7786,
    "gym_timing_single_env": 4836.4002
  }
]



if __name__ == "__main__":

    # Create a row of subplots
    fig, axes = plt.subplots(1, len(data), figsize=(8 * len(data), 7), sharey=True)
    time_per_gpu_env = {}

    for ax, env_data in zip(axes, data):
        env = env_data["env"]
        n_envs = env_data["n_envs"]
        gpu = env_data["gpu_timings"]
        cpu = env_data["cpu_timing_single_env"]
        gym = env_data["gym_timing_single_env"]

        # Plot GPU timings vs number of envs
        gpu_line = ax.plot(gpu, n_envs, marker='o', linestyle='-', color=colors['blue'], label='GPU')

        # Vertical lines for CPU and Gym single-env timings
        cpu_vline = ax.axvline(cpu, color=colors['purple'], linestyle='--', label='CPU (1 env)')
        gym_vline = ax.axvline(gym, color=colors['orange'], linestyle='-.', label='Gym (1 env)')
        per_env_gpu_time = gpu[-1] / n_envs[-1]
        gpu_single_vline = ax.axvline(per_env_gpu_time, color=colors['blue'], linestyle='--', label='GPU (per env)')

        time_per_gpu_env[env] = per_env_gpu_time

        # Labels and title
        ax.set_title(env_to_name[env])
        ax.grid(True, which='both', axis='both', linestyle=':', linewidth=0.5)
        # ax.legend()

    # Common y-label
    axes[0].set_ylabel('Num. Environments')
    lines = [gpu_line, cpu_vline, gym_vline, gpu_single_vline]
    labels = ['GPU', 'CPU (1 env)', 'Gym (1 env)', 'GPU (per env)']

    # Customize legend to use square markers
    leg = plt.figlegend(lines, labels, loc=(0.1, 0.4))
    leg.set_alpha(0.95)

    fig.supxlabel('Time for 10^7 steps (seconds)')

    fig.tight_layout()
    plt.show()

    save_plot_to = Path(ROOT_DIR, 'results', f'timings.pdf')

    fig.savefig(save_plot_to, bbox_inches='tight', dpi=100)
    print(f"Saved figure to {save_plot_to}")
