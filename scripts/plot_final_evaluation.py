import importlib
import pickle
from pathlib import Path

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.stats import sem


from definitions import ROOT_DIR

rc('font', **{'family': 'serif', 'serif': ['cmr10']})
rc('axes', unicode_minus=False)

# rc('text', usetex=True)

colors = {
    'pink': '#ff96b6',
    'red': '#df5b5d',
    'orange': '#DD8453',
    'yellow': '#f8de7c',
    'green': '#3FC57F',
    'cyan': '#48dbe5',
    'blue': '#3180df',
    'purple': '#9d79cf',
    'brown': '#886a2c',
    'white': '#ffffff',
    'light gray': '#d5d5d5',
    'dark gray': '#666666',
    'black': '#000000'
}

lengends = {
    'RNN': 'blue',
    'Memoryless + Depth 3': 'red',
    'Memoryless + Depth 5': 'green',
    'Memoryless + Depth 7': 'yellow'
}

env_name_to_title = {
    'rocksample_15_15': 'RockSample (15, 15)',
    'rocksample_11_11': 'RockSample (11, 11)',
    'pocman': 'Pocman',
    'battleship_10': 'Battleship 10x10',
    'battleship_5': 'Battleship 5x5',
    'cheese.95': 'Cheese',
    'hallway': 'Hallway',
    'heavenhell': 'Heavenhell',
    'network': 'Network',
    'paint': 'Paint',
    'shuttle': 'Shuttle',
    'tiger-alt-start': 'Tiger',
    'tmaze_5': 'T-maze'
}

env_name_to_x_upper_lim = {
    '4x3': 1e6,
    'cheese.95': 1e6,
    'hallway': 2e6,
    'network': 1e6,
    'paint': 1e6,
    'tiger-alt-start': 1e6,
    'tmaze_5': 2e6
}

def plot_reses(all_reses: list[tuple], n_rows: int = 2,
               individual_runs: bool = False):
    plt.rcParams.update({'font.size': 32})

    # check to see that all our envs are the same across all reses.
    all_envs = [set(x['envs']) for _, x, _ in all_reses]
    for envs in all_envs:
        assert envs == all_envs[0]

    envs = list(sorted(all_envs[0]))

    n_rows = min(n_rows, len(envs))
    n_cols = max((len(envs) + 1) // n_rows, 1) if len(envs) > 1 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 20))
    bar_width = 0.2
    hidden_sizes = [4, 8, 16, 32, 64, 128]

    for k, (study_name, res, color) in enumerate(all_reses):
        scores = res['final_scores']
        hidden_size = res['hidden_size']
        print(f"Hidden size: {hidden_size}")
        if isinstance(scores, list):
            mean_over_steps = [score.mean(axis=1)[..., 0] for score in scores]
            mean = [m.mean(axis=-1) for m in mean_over_steps]
            std_err = [sem(m, axis=-1) for m in mean_over_steps]
        else:
            # take mean over both
            mean_over_steps = scores.mean(axis=1)
            # if 'lambda' in study_name:
            #     mean_over_steps = mean_over_steps[..., 5:, :]
            mean = mean_over_steps.mean(axis=-2)
            std_err = sem(mean_over_steps, axis=-2)

        for i, env in enumerate(envs):
            row = i // n_cols
            col = i % n_cols

            env_idx = res['envs'].index(env)
            if isinstance(mean, list):
                env_mean, env_std_err = mean[env_idx], std_err[env_idx]
                n_seeds = mean_over_steps[0].shape[-1]
            else:
                env_mean, env_std_err = mean[..., env_idx], std_err[..., env_idx]
                n_seeds = mean_over_steps.shape[-2]

            if len(envs) == 1:
                ax = axes
            else:
                ax = axes[row, col] if n_cols > 1 else axes[i]

            # ax.plot(hidden_sizes, env_mean, label=study_name, color=colors[color])
            # if individual_runs:
            #     for j in range(mean_over_steps.shape[-2]):
            #         alpha = 1 / mean_over_steps.shape[-2]
            #         m = mean_over_steps[..., j, env_idx] if isinstance(mean_over_steps, np.ndarray) else mean_over_steps[env_idx][..., j]
            #         ax.plot(hidden_sizes, m, color=colors[color], alpha=alpha)
            # else:
            #     ax.fill_between(hidden_sizes, env_mean - env_std_err, env_mean + env_std_err,
            #                     color=colors[color], alpha=0.35)

            bar_positions = hidden_size + k * bar_width
            ax.bar(bar_positions, env_mean, yerr=env_std_err, color=colors[color], alpha=0.7, capsize=5)
            ax.errorbar(bar_positions, env_mean, yerr=env_std_err, fmt='o', color=colors[color], label=study_name, capsize=5, elinewidth=2, markeredgewidth=2)
        
            ax.set_title(env_name_to_title.get(env, env))
            # ax.set_ylim(bottom=0)  # Ensure the y-axis starts from 0
            ax.set_xticks(np.array(hidden_sizes) + bar_width * (len(all_reses) - 1) / 2)
            ax.set_xticklabels(hidden_sizes)
            ax.locator_params(axis='x', nbins=6)
            ax.locator_params(axis='y', nbins=3)
            ax.spines[['right', 'top']].set_visible(False)

    fig.supxlabel('Hidden size')
    fig.supylabel(f'Final evaluation ({n_seeds} runs)')
    fig.legend([plt.Line2D([0], [0], color=colors[lengends[key]], lw=4) for key in lengends.keys()],
               [key for key in lengends.keys()],
               loc='lower right')
    #
    fig.tight_layout()

    plt.show()
    return fig, axes


def find_file_in_dir(file_name: str, base_dir: Path) -> Path:
    for path in base_dir.rglob('*'):
        if file_name in str(path):
            return path

if __name__ == "__main__":
    env_name = 'reacher'

    # normal
    study_paths = [
        # ('$\lambda$-discrepancy + PPO', Path(ROOT_DIR, 'results', f'{env_name}_LD_ppo'), 'green'),
        # ('PPO', Path(ROOT_DIR, 'results', f'{env_name}_rnn_ppo'), 'blue'),
        # ('Memoryless PPO', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_ppo'), 'dark gray'),
        # ('PPO', Path(ROOT_DIR, 'results', f'{env_name}_rnn_ppo_best'), 'blue'),
        # ('Memoryless PPO', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_ppo_best'), 'dark gray'),

        # depth
        ('RNN + Hidden Size 4', Path(ROOT_DIR, 'saved_results', 'rnn_hidden_size', f'{env_name}_rnn_hidden4_ppo_best'), 'blue'),
        ('RNN + Hidden Size 8', Path(ROOT_DIR, 'saved_results', 'rnn_hidden_size', f'{env_name}_rnn_hidden8_ppo_best'), 'blue'),
        ('RNN + Hidden Size 16', Path(ROOT_DIR, 'saved_results', 'rnn_hidden_size', f'{env_name}_rnn_hidden16_ppo_best'), 'blue'),
        ('RNN + Hidden Size 32', Path(ROOT_DIR, 'saved_results', 'rnn_hidden_size', f'{env_name}_rnn_hidden32_ppo_best'), 'blue'),
        ('RNN + Hidden Size 64', Path(ROOT_DIR, 'saved_results', 'rnn_hidden_size', f'{env_name}_rnn_hidden64_ppo_best'), 'blue'),
        ('RNN + Hidden Size 128', Path(ROOT_DIR, 'saved_results', 'rnn_hidden_size', f'{env_name}_rnn_hidden128_ppo_best'), 'blue'),
        ('Memoryless + Hidden Size 4', Path(ROOT_DIR, 'saved_results', 'mlp_depth3_hidden_size', f'{env_name}_memoryless_hidden4_ppo_best'), 'red'),
        ('Memoryless + Hidden Size 8', Path(ROOT_DIR, 'saved_results', 'mlp_depth3_hidden_size', f'{env_name}_memoryless_hidden8_ppo_best'), 'red'),
        ('Memoryless + Hidden Size 16', Path(ROOT_DIR, 'saved_results', 'mlp_depth3_hidden_size', f'{env_name}_memoryless_hidden16_ppo_best'), 'red'),
        ('Memoryless + Hidden Size 32', Path(ROOT_DIR, 'saved_results', 'mlp_depth3_hidden_size', f'{env_name}_memoryless_hidden32_ppo_best'), 'red'),
        ('Memoryless + Hidden Size 64', Path(ROOT_DIR, 'saved_results', 'mlp_depth3_hidden_size', f'{env_name}_memoryless_hidden64_ppo_best'), 'red'),
        ('Memoryless + Hidden Size 128', Path(ROOT_DIR, 'saved_results', 'mlp_depth3_hidden_size', f'{env_name}_memoryless_hidden128_ppo_best'), 'red'),
        ('Memoryless + Hidden Size 4', Path(ROOT_DIR, 'saved_results', 'mlp_depth5_hidden_size', f'{env_name}_memoryless_hidden4_ppo_best'), 'green'),
        ('Memoryless + Hidden Size 8', Path(ROOT_DIR, 'saved_results', 'mlp_depth5_hidden_size', f'{env_name}_memoryless_hidden8_ppo_best'), 'green'),
        ('Memoryless + Hidden Size 16', Path(ROOT_DIR, 'saved_results', 'mlp_depth5_hidden_size', f'{env_name}_memoryless_hidden16_ppo_best'), 'green'),
        ('Memoryless + Hidden Size 32', Path(ROOT_DIR, 'saved_results', 'mlp_depth5_hidden_size', f'{env_name}_memoryless_hidden32_ppo_best'), 'green'),
        ('Memoryless + Hidden Size 64', Path(ROOT_DIR, 'saved_results', 'mlp_depth5_hidden_size', f'{env_name}_memoryless_hidden64_ppo_best'), 'green'),
        ('Memoryless + Hidden Size 128', Path(ROOT_DIR, 'saved_results', 'mlp_depth5_hidden_size', f'{env_name}_memoryless_hidden128_ppo_best'), 'green'),
        ('Memoryless + Hidden Size 4', Path(ROOT_DIR, 'saved_results', 'mlp_depth7_hidden_size', f'{env_name}_memoryless_hidden4_ppo_best'), 'yellow'),
        ('Memoryless + Hidden Size 8', Path(ROOT_DIR, 'saved_results', 'mlp_depth7_hidden_size', f'{env_name}_memoryless_hidden8_ppo_best'), 'yellow'),
        ('Memoryless + Hidden Size 16', Path(ROOT_DIR, 'saved_results', 'mlp_depth7_hidden_size', f'{env_name}_memoryless_hidden16_ppo_best'), 'yellow'),
        ('Memoryless + Hidden Size 32', Path(ROOT_DIR, 'saved_results', 'mlp_depth7_hidden_size', f'{env_name}_memoryless_hidden32_ppo_best'), 'yellow'),
        ('Memoryless + Hidden Size 64', Path(ROOT_DIR, 'saved_results', 'mlp_depth7_hidden_size', f'{env_name}_memoryless_hidden64_ppo_best'), 'yellow'),
        ('Memoryless + Hidden Size 128', Path(ROOT_DIR, 'saved_results', 'mlp_depth7_hidden_size', f'{env_name}_memoryless_hidden128_ppo_best'), 'yellow'),
    ]

    hyperparam_type = 'per_env'  # (all_env | per_env)
    plot_name = f'{env_name}_{hyperparam_type}'

    if study_paths[0][1].stem.endswith('best'):
        plot_name += '_best'

    envs = None

    all_reses = []

    for name, study_path, color in study_paths:
        if hyperparam_type == 'all_env':
            fname = "best_hyperparam_res.pkl"
            if name == 'PPO Markov':
                fname = 'best_hyperparam_res_F_split.pkl'
        elif hyperparam_type == 'per_env':
            fname = "best_hyperparam_per_env_res.pkl"

        with open(study_path / fname, "rb") as f:
            best_res = pickle.load(f)

        print(f"Best hyperparams for {name}: {best_res['hyperparams']}")
        all_reses.append((name, best_res, color))
        
    fig, axes = plot_reses(all_reses, individual_runs=False, n_rows=2)

    save_plot_to = Path(ROOT_DIR, 'graphs', f'{plot_name}_hidden_size.jpg')

    fig.savefig(save_plot_to, bbox_inches='tight')
    print(f"Saved figure to {save_plot_to}")