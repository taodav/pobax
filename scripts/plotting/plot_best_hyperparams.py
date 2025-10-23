import importlib
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from scipy.stats import sem

from pobax.utils.plot import mean_confidence_interval, colors, smoothen

from pobax.definitions import PROJECT_ROOT_DIR

rc('font', **{'family': 'serif', 'serif': ['cmr10']})
rc('axes', unicode_minus=False)

# rc('text', usetex=True)

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

fully_observable_to_base = {
    'Navix-DMLab-Maze-F-03-v0': 'Navix-DMLab-Maze-03-v0',
    'Navix-DMLab-Maze-F-02-v0': 'Navix-DMLab-Maze-02-v0',
    'Navix-DMLab-Maze-F-01-v0': 'Navix-DMLab-Maze-01-v0',
    'Navix-Annas-Maze-F-v0': 'Navix-Annas-Maze-v0',
}

def plot_reses(all_reses: list[tuple], n_rows: int = 2,
               individual_runs: bool = False,
               plot_title: str = None,
               discounted: bool = False,
               ylims: Tuple[float, float] = None):
    plt.rcParams.update({'font.size': 24})
    
    # check to see that all our envs are the same across all reses.
    for _, x, _ in all_reses:
        for i in range(len(x['envs'])):
            if x['envs'][i].endswith('pixels'):
                x['envs'][i] = x['envs'][i][:-7]
                print(x['envs'][i])
    all_envs = [set(x['envs']) for _, x, _ in all_reses]
    for envs in all_envs:
        assert envs == all_envs[0]

    envs = list(sorted(all_envs[0]))

    n_rows = min(n_rows, len(envs))
    n_cols = max((len(envs) + 1) // n_rows, 1) if len(envs) > 1 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))

    for k, (study_name, res, color) in enumerate(all_reses):
        scores = res['scores']
        if isinstance(scores, list):
            mean_over_steps = [score.mean(axis=1)[..., 0] for score in scores]
            mean = [m.mean(axis=-1) for m in mean_over_steps]
            std_err = [sem(m, axis=-1) for m in mean_over_steps]
        elif isinstance(scores, dict):
            mean, std_err = {}, {}
            n_seeds = None
            for k, v in scores.items():
                mean_over_steps = v.mean(axis=1)
                if n_seeds is None:
                    n_seeds = mean_over_steps.shape[-1]

                m, c = mean_confidence_interval(mean_over_steps, axis=-1)
                mean[k] = m
                std_err[k] = c
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
            elif isinstance(mean, dict):
                env_mean, env_std_err = mean[env], std_err[env]
            else:
                env_mean, env_std_err = mean[..., env_idx], std_err[..., env_idx]
                n_seeds = mean_over_steps.shape[-2]

            # env_mean, env_std_err = env_mean[10:], env_std_err[10:]
            x_axis_multiplier = res['step_multiplier'][env_idx]

            if len(envs) == 1:
                ax = axes
            else:
                ax = axes[row, col] if n_cols > 1 else axes[i]
            x = np.arange(env_mean.shape[0]) * x_axis_multiplier
            x_upper_lim = env_name_to_x_upper_lim.get(env, None)

            ax.plot(x, env_mean, label=study_name, color=colors[color])
            if individual_runs:
                # -1 index is seeds.
                # mean_over_steps = mean_over_steps[..., 20:30]
                alpha = (mean_over_steps.shape[-1] - mean_over_steps.shape[-1] // 3) / mean_over_steps.shape[-1]
                for j in range(mean_over_steps.shape[-1]):
                    # m = mean_over_steps[..., j, env_idx] if isinstance(mean_over_steps, np.ndarray) else mean_over_steps[env_idx][..., j]
                    m = mean_over_steps[..., j]
                    ax.plot(x, m, color=colors[color], alpha=alpha)
            else:
                ax.fill_between(x, env_mean - env_std_err, env_mean + env_std_err,
                                color=colors[color], alpha=0.35)
            if plot_title is None:
                plot_title = env_name_to_title.get(env, env)
            ax.set_title(plot_title)
            if x_upper_lim is not None:
                ax.set_xlim(right=x_upper_lim)
            if ylims is not None:
                ax.set_ylim(bottom=ylims[0], top=ylims[1])
            # ax.margins(x=0.015)
            ax.locator_params(axis='x', nbins=3, min_n_ticks=3)
            ax.locator_params(axis='y', nbins=3)
            ax.spines[['right', 'top']].set_visible(False)

    # Customize legend to use square markers
    # after all plotting loops
    legend = fig.legend(ncol=1, bbox_to_anchor=(1.01, 0.8), prop={'size': 16})

    # # Change line in legend to square
    # for line in legend.get_lines():
    #     line.set_marker('s')
    #     line.set_markerfacecolor(line.get_color())
    #     line.set_linestyle('')
    #     line.set_markersize(20)  # Increase the marker size

    fig.supxlabel('Environment steps')
    if discounted:
        fig.supylabel(f'Online discounted returns ({n_seeds} runs)')
    else:
        fig.supylabel(f'Online returns ({n_seeds} runs)')
    #
    fig.tight_layout()
    plt.subplots_adjust(right=0.8)

    plt.show()
    return fig, axes


def get_total_steps_multiplier(saved_steps: int, hparam_path: Path):
    spec = importlib.util.spec_from_file_location('temp', hparam_path)
    var_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(var_module)
    all_list_hparams = getattr(var_module, 'hparams')['args']

    steps_multipliers = []
    for hparams in all_list_hparams:
        assert 'total_steps' in hparams
        total_steps = hparams['total_steps']
        steps_multiplier = total_steps // saved_steps
        steps_multipliers.append(steps_multiplier)

    assert all(m == steps_multipliers[0] for m in steps_multipliers)
    return steps_multipliers[0]


def find_file_in_dir(file_name: str, base_dir: Path) -> Path:
    for path in base_dir.rglob('*'):
        if file_name in str(path):
            return path


if __name__ == "__main__":

    hyperparam_type = 'per_env'  # (all_env | per_env)

    discounted = True
    env_name = 'annas_maze'
    super_dir = 'new_pobax_res/annas_maze'
    best = False

    best_str = '_best' if best else ''
    # super_dir += best_str

    ylims = None

    plot_name = f'{env_name}_{hyperparam_type}'

    # normal
    study_paths = [
        ('RNN', Path(PROJECT_ROOT_DIR, 'results', super_dir, f'{env_name}_ppo'), 'purple'),
        ('LD', Path(PROJECT_ROOT_DIR, 'results', super_dir, f'{env_name}_ppo_LD'), 'blue'),
        ('Memoryless', Path(PROJECT_ROOT_DIR, 'results', super_dir, f'{env_name}_ppo_memoryless{best_str}'), 'dark gray'),
        # ('STATE', Path(ROOT_DIR, 'results', super_dir, f'{env_name}_ppo_perfect_memory_memoryless{best_str}'), 'green'),
        ('STATE', Path(PROJECT_ROOT_DIR, 'results', super_dir, f'{env_name}_ppo_perfect_memory{best_str}'), 'green'),
        ('TRANSFORMER', Path(PROJECT_ROOT_DIR, 'results', super_dir, f'{env_name}_transformer{best_str}'), 'yellow'),
    ]


    # best
    # study_paths = [
    #     ('PPO + RNN + LD', Path(ROOT_DIR, 'results', f'{env_name}_LD_ppo_best'), 'green'),
    #     ('PPO + RNN', Path(ROOT_DIR, 'results', f'{env_name}_ppo_best'), 'blue'),
    #     # ('Memoryless PPO', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_ppo_best'), 'dark gray'),
    # ]


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
            if discounted:
                fname = "best_hyperparam_per_env_res_discounted.pkl"

        with open(study_path / fname, "rb") as f:
            best_res = pickle.load(f)
            new_envs = []
            for env in best_res['envs']:
                if env in fully_observable_to_base:
                    if fully_observable_to_base[env] in best_res['hyperparams']:
                        best_res['hyperparams'][fully_observable_to_base[env]] = best_res['hyperparams'][env]
                        del best_res['hyperparams'][env]
                    new_envs.append(fully_observable_to_base[env])
                else:
                    new_envs.append(env)
            best_res['envs'] = new_envs

            assert len(best_res['envs']) == 1
            env_name = best_res['envs'][0]
            if '-F-' in env_name:
                first_res = all_reses[0][1]
                best_res['envs'] = first_res['envs']

        all_reses.append((name, best_res, color))

        if isinstance(best_res['scores'], dict):
            denom = list(best_res['scores'].values())[0].shape[0]
        else:
            denom = best_res['scores'].shape[0]

        if 'all_hyperparams' in best_res:
            step_multiplier = best_res['all_hyperparams']['total_steps'] // denom
        elif 'total_steps' in best_res['hyperparams']:
            step_multiplier = best_res['hyperparams']['total_steps'] // denom
        else:
            raise NotImplementedError("Missing total steps")

            # hyperparams_dir = study_path.parent.parent / 'scripts' / 'hyperparams'
            # study_hparam_filename = study_path.stem + '.py'
            # hyperparam_path = find_file_in_dir(study_hparam_filename, hyperparams_dir)
            # step_multiplier = get_total_steps_multiplier(best_res['scores'].shape[0], hyperparam_path)
        best_res['step_multiplier'] = [step_multiplier] * len(best_res['envs'])

    fig, axes = plot_reses(all_reses, individual_runs=False, n_rows=1, plot_title=plot_name,
                           discounted=discounted, ylims=ylims)

    discount_str = '_discounted' if discounted else ''
    save_plot_to = Path(ROOT_DIR, 'results', f'{plot_name}{discount_str}.pdf')

    fig.savefig(save_plot_to, bbox_inches='tight')
    print(f"Saved figure to {save_plot_to}")

