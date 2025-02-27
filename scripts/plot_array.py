from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from matplotlib import rc
import numpy as np

from pobax.utils.plot import mean_confidence_interval, colors

from definitions import ROOT_DIR

rc('font', **{'family': 'serif', 'serif': ['cmr10'], 'size': 32})
rc('axes', unicode_minus=False)

all_paths = {
    'rocksample_11_11': {
        'position': (0, 1),
        'best': True,
        'super_dir': 'rocksample_11_11_best',
        'state_version': 'recurrent',
        'title': 'RockSample(11, 11)'
    },
    'rocksample_15_15': {
        'position': (0, 2),
        'best': True,
        'super_dir': 'rocksample_15_15_best',
        'state_version': 'recurrent',
        'title': 'RockSample(15, 15)'
    },
    'battleship_10': {
        'position': (0, 3),
        'best': True,
        'state_version': 'battleship',
        'super_dir': 'battleship_best',
        'title': 'BattleShip'
},
    # 'pocman': {
    #     'position': (0, 1),
    #     'best': True,
    #     'super_dir': 'battleship_best'
    # },
    'walker_v': {
        'position': (1, 0),
        'best': False,
        'super_dir': 'walker_v',
        'state_version': 'recurrent',
        'title': 'Walker (Vel. Only)'
    },
    'halfcheetah_v': {
        'position': (1, 1),
        'best': False,
        'super_dir': 'halfcheetah_v',
        'state_version': 'recurrent',
        'title': 'HalfCheetah (Vel. Only)'
    },
    'navix_01': {
        'position': (1, 2),
        'best': True,
        'super_dir': 'navix_01_best',
        'state_version': 'memoryless',
        'title': 'DMLab Minigrid (01)'
    },
    'navix_02': {
        'position': (1, 3),
        'best': True,
        'super_dir': 'navix_02_best',
        'state_version': 'memoryless',
        'title': 'DMLab Minigrid (02)'
    },
    'ant_pixels': {
        'position': (2, 0),
        'best': False,
        'super_dir': 'ant_pixels',
        'state_version': 'memoryless',
        'title': 'Visual Ant'
    },
    'halfcheetah_pixels': {
        'position': (2, 1),
        'best': False,
        'super_dir': 'halfcheetah_pixels',
        'state_version': 'memoryless',
        'title': 'Visual HalfCheetah'
    },
    'craftax': {
        'position': (2, 2),
        'best': False,
        'super_dir': 'craftax',
        'state_version': 'memoryless',
        'title': 'No-Inv. Craftax'
    },
}

def generate_study_paths(paths: dict):
    all_study_paths = {}
    results_path = Path(ROOT_DIR) / 'results'

    for name, info in paths.items():
        super_dir = results_path / info['super_dir']
        best_suffix = '_best' if info['best'] else ''

        paths = [
                ('RNN', super_dir / f"{name}_ppo{best_suffix}", "purple"),
                ('RNN + LD', super_dir / f"{name}_ppo_LD{best_suffix}", 'blue'),
                ('OBSERVATION', super_dir / f"{name}_ppo_memoryless{best_suffix}", 'dark gray'),
                ('TRANSFORMER', super_dir / f"{name}_transformer{best_suffix}", 'cyan'),
            ]

        # ORDER MATTERS HERE! Assume all the -F- runs are after the rest.
        if info['state_version'] == 'memoryless':
            paths.append(('FULL STATE', super_dir / f"{name}_ppo_perfect_memory_memoryless{best_suffix}", 'dark green'))
        elif info['state_version'] == 'recurrent':
            paths.append(('FULL STATE', super_dir / f"{name}_ppo_perfect_memory{best_suffix}", 'green'))

        curr_path = {
            **info,
            'paths': paths
        }
        all_study_paths[name] = curr_path
    return all_study_paths


def plot_array(all_res: dict,
               discounted: bool = False):
    # first figure out what size array we want
    n_rows, n_cols = (np.array([res['position'] for res in all_res.values()]) + 1).max(axis=0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10 * n_rows, 5 * n_cols))

    lines, labels = [], []

    for name, info in all_res.items():
        row, col = info['position']
        assert not ((row == (n_rows - 1)) and (col == (n_cols - 1))), "Legend is blocking a plot!"

        # set up our axis
        ax = axes[row, col]
        ax.set_title(info['title'])

        ax.locator_params(axis='x', nbins=3, min_n_ticks=3)
        ax.locator_params(axis='y', nbins=3)
        ax.spines[['right', 'top']].set_visible(False)

        for algo_name, res, color in info['res']:

            if res is None:
                continue

            # take mean over num_steps
            mean_over_steps = res['scores'].mean(axis=1)

            # make sure we only have one env
            assert mean_over_steps.shape[-1] == 1
            mean_over_steps = mean_over_steps.mean(axis=-1)

            # now get our actual mean and confidence interval over axis=-1
            mean, conf = mean_confidence_interval(mean_over_steps, axis=-1)

            x = np.arange(mean.shape[0]) * res['step_multiplier'][0]

            ax.plot(x, mean, label=algo_name, color=colors[color])
            ax.fill_between(x, mean - conf, mean + conf, color=colors[color], alpha=0.35)

        if not lines:
            lines, labels = ax.get_legend_handles_labels()

    # We add extraneous plots here.
    # plot battleship optimal policy
    info = all_res['battleship_10']
    row, col = info['position']
    ax = axes[row, col]
    left, right = ax.get_xlim()
    mean, conf = 46, 0.54925
    means = np.array([mean, mean])
    ax.hlines(46.0, left, right, linestyles='dashed', colors=colors['green'])
    ax.fill_between(np.array([left, right]), means - conf, means + conf, color=colors[color], alpha=0.35)

    # we clear the final axis
    axes[-1, -1].set_axis_off()

    # first set the handles symbols to something else
    for l in lines:
        l.set_marker('s')
        l.set_markersize(16)
        l.set_linestyle('None')

    plt.figlegend(lines, labels, loc=(0.78, 0.13))

    fig.supxlabel('Environment steps')
    if discounted:
        fig.supylabel(f'Online discounted returns')
    else:
        fig.supylabel(f'Online returns')

    fig.tight_layout()

    plt.show()

    return fig, axes



if __name__ == "__main__":
    discounted = True
    all_study_settings = generate_study_paths(all_paths)

    all_res = {}
    for name, info in all_study_settings.items():
        paths = info['paths']
        del info['paths']
        env_res = {
            **info,
            'res': []
        }

        for algo_name, study_path, color in paths:
            if discounted:
                fpath = study_path / "best_hyperparam_per_env_res_discounted.pkl"
            else:
                fpath = study_path / "best_hyperparam_per_env_res.pkl"
                if not fpath.exists():
                    fpath = study_path / "best_hyperparam_per_env_res_undiscounted.pkl"

            best_res = None
            try:
                with open(fpath, "rb") as f:
                    best_res = pickle.load(f)
            except FileNotFoundError as e:
                print(f"MISSING: {fpath.parent}")

            if best_res is not None:
                if 'all_hyperparams' in best_res:
                    step_multiplier = best_res['all_hyperparams']['total_steps'] // best_res['scores'].shape[0]
                elif 'total_steps' in best_res['hyperparams']:
                    # Missing hyperparams == all_hyperparams
                    step_multiplier = best_res['hyperparams']['total_steps'] // best_res['scores'].shape[0]
                else:
                    raise NotImplementedError("Missing total steps")

                # WE DEAL WITH MASKED MUJOCO fully observable envs HERE.
                # we check the env name of the first res in env_res['res']
                assert len(best_res['envs']) == 1
                env_name = best_res['envs'][0]
                if '-F-' in env_name:
                    first_res = env_res['res'][0][1]
                    best_res['envs'] = first_res['envs']

                best_res['step_multiplier'] = [step_multiplier] * len(best_res['envs'])

            env_res['res'].append((algo_name, best_res, color))

        all_res[name] = env_res

    fig, axes = plot_array(all_res, discounted=discounted)


    discount_str = '_discounted' if discounted else ''
    save_plot_to = Path(ROOT_DIR, 'results', f'all_envs{discount_str}.pdf')

    fig.savefig(save_plot_to, bbox_inches='tight')
    print(f"Saved figure to {save_plot_to}")