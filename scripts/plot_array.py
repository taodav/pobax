from pathlib import Path
import pickle

import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.lines import Line2D
import numpy as np

from pobax.utils.plot import mean_confidence_interval, colors, smoothen

from definitions import ROOT_DIR

rc('font', **{'family': 'serif', 'serif': ['cmr10'], 'size': 32})
rc('axes', unicode_minus=False)

walker_hsize_paths = {
    32: {
        'env': 'walker_v',
        'sweep_var': 'hsize',
        'position': (0, 0),
        'state_version': 'recurrent',
        'title': 'Network Width = 32',
        'discounted': False,
        'range': (0, 2500)
    },
    64: {
        'env': 'walker_v',
        'sweep_var': 'hsize',
        'position': (0, 1),
        'state_version': 'recurrent',
        'title': 'Network Width = 64',
        'discounted': False,
        'range': (0, 2500)
    },
    256: {
        'env': 'walker_v',
        'sweep_var': 'hsize',
        'position': (0, 2),
        'state_version': 'recurrent',
        'title': 'Network Width = 256',
        'discounted': False,
        'range': (0, 2500)
    },

}

navix_nenvs_paths = {
    64: {
        'env': 'navix_01',
        'sweep_var': 'nenvs',
        'position': (0, 0),
        'state_version': 'recurrent',
        'title': 'Num. Envs. = 64',
        'discounted': True,
        'range': (0, 0.75)
    },
    256: {
        'env': 'navix_01',
        'sweep_var': 'nenvs',
        'position': (0, 1),
        'state_version': 'recurrent',
        'title': 'Num. Envs. = 256',
        'discounted': True,
        'range': (0, 0.75)
    },

}

masked_mujoco_paths = {
    'ant_p': {
        'position': (0, 0),
        'best': True,
        'super_dir': 'masked_mujoco_best/ant_p_best',
        'state_version': 'memoryless',
        'title': 'Ant (Pos. Only)',
        'discounted': False,
    },
    'ant_v': {
        'position': (1, 0),
        'best': True,
        'super_dir': 'masked_mujoco_best/ant_v_best',
        'state_version': 'memoryless',
        'title': 'Ant (Vel. Only)',
        'discounted': False,
    },
    'hopper_p': {
        'position': (0, 1),
        'best': True,
        'super_dir': 'masked_mujoco_best/hopper_p_best',
        'state_version': 'memoryless',
        'title': 'Hopper (Pos. Only)',
        'discounted': False,
    },
    'hopper_v': {
        'position': (1, 1),
        'best': True,
        'super_dir': 'masked_mujoco_best/hopper_v_best',
        'state_version': 'memoryless',
        'title': 'Hopper (Vel. Only)',
        'discounted': False,
    },
    'halfcheetah_p': {
        'position': (0, 2),
        'best': True,
        'super_dir': 'masked_mujoco_best/halfcheetah_p_best',
        'state_version': 'memoryless',
        'title': 'HalfCheetah (Pos. Only)',
        'discounted': False,
    },
    'halfcheetah_v': {
        'position': (1, 2),
        'best': True,
        'super_dir': 'masked_mujoco_best/halfcheetah_v_best',
        'state_version': 'memoryless',
        'title': 'HalfCheetah (Vel. Only)',
        'discounted': False,
    },
    'walker_p': {
        'position': (0, 3),
        'best': True,
        'super_dir': 'masked_mujoco_best/walker_p_best',
        'state_version': 'memoryless',
        'title': 'Walker (Pos. Only)',
        'discounted': False,
    },
    'walker_v': {
        'position': (1, 3),
        'best': True,
        'super_dir': 'masked_mujoco_best/walker_v_best',
        'state_version': 'memoryless',
        'title': 'Walker (Vel. Only)',
        'discounted': False,
    },

}

all_paths = {
    'tmaze_10': {
        'position': (0, 0),
        'best': True,
        'super_dir': 'tmaze_best',
        'state_version': 'memoryless',
        'title': 'T-Maze 10',
        'discounted': True
    },
    'rocksample_11_11': {
        'position': (0, 1),
        'best': True,
        'super_dir': 'rocksample_11_11_best',
        'state_version': 'recurrent',
        'title': 'RockSample(11, 11)',
        'discounted': True
    },
    'rocksample_15_15': {
        'position': (0, 2),
        'best': True,
        'super_dir': 'rocksample_15_15_best',
        'state_version': 'recurrent',
        'title': 'RockSample(15, 15)',
        'discounted': True,
    },
    'battleship_10': {
        'position': (0, 3),
        'best': True,
        'state_version': 'battleship',
        'super_dir': 'battleship_best',
        'title': 'BattleShip',
        'discounted': False,
    },
    # 'pocman': {
    #     'position': (0, 1),
    #     'best': True,
    #     'super_dir': 'pocman_best'
    #     'discounted': False,
    # },
    'walker_v': {
        'position': (1, 0),
        'best': True,
        'super_dir': 'walker_v_best',
        'state_version': 'recurrent',
        'title': 'Walker (Vel. Only)',
        'discounted': False,
        'smoothen_curve': True,
    },
    'halfcheetah_v': {
        'position': (1, 1),
        'best': True,
        'super_dir': 'halfcheetah_v_best',
        'state_version': 'recurrent',
        'title': 'HalfCheetah (Vel. Only)',
        'discounted': False,
        'smoothen_curve': True,
    },
    'navix_01': {
        'position': (1, 2),
        'best': True,
        'super_dir': 'navix_01_best',
        'state_version': 'memoryless',
        'title': 'DMLab Minigrid (01)',
        'discounted': True,
    },
    'navix_02': {
        'position': (1, 3),
        'best': True,
        'super_dir': 'navix_02_best',
        'state_version': 'memoryless',
        'title': 'DMLab Minigrid (02)',
        'discounted': True,
    },
    'ant_pixels': {
        'position': (2, 0),
        'best': True,
        'super_dir': 'ant_pixels_best',
        'state_version': 'memoryless',
        'title': 'Visual Ant',
        'discounted': False,
        'smoothen_curve': True,
        'range': (-1000, 2000)
    },
    'halfcheetah_pixels': {
        'position': (2, 1),
        'best': True,
        'super_dir': 'halfcheetah_pixels_best',
        'state_version': 'memoryless',
        'title': 'Visual HalfCheetah',
        'discounted': False,
    },
    'craftax': {
        'position': (2, 2),
        'best': True,
        'super_dir': 'craftax_best',
        'state_version': 'recurrent',
        'title': 'No-Inv. Crafter',
        'discounted': False,
    },
}

def generate_ablation_paths(paths: dict):
    all_study_paths = {}
    for var_val, info in paths.items():
        sweep_var, name = info['sweep_var'], info['env']
        paths = [
            ('RNN', Path(ROOT_DIR, 'results', f'{name}_{sweep_var}_sweep/{name}_ppo_{sweep_var}_sweep',
                         f'{name}_ppo_{sweep_var}_sweep_{sweep_var}_{var_val}'), 'purple'),
            ('Memoryless', Path(ROOT_DIR, 'results', f'{name}_{sweep_var}_sweep/{name}_ppo_memoryless_{sweep_var}_sweep',
                  f'{name}_ppo_memoryless_{sweep_var}_sweep_{sweep_var}_{var_val}'), 'dark gray'),
        ]
        # ORDER MATTERS HERE! Assume all the -F- runs are after the rest.
        if info['state_version'] == 'memoryless':
            paths.append(('STATE', Path(ROOT_DIR, 'results', f'{name}_{sweep_var}_sweep/{name}_ppo_perfect_memory_{sweep_var}_sweep',
                                        f'{name}_ppo_perfect_memory_memoryless_{sweep_var}_sweep_{sweep_var}_{var_val}'), 'green'))
        elif info['state_version'] == 'recurrent':
            paths.append(('STATE', Path(ROOT_DIR, 'results', f'{name}_{sweep_var}_sweep/{name}_ppo_perfect_memory_{sweep_var}_sweep',
                                        f'{name}_ppo_perfect_memory_{sweep_var}_sweep_{sweep_var}_{var_val}'), 'green'))
        curr_path = {
            **info,
            'paths': paths
        }
        all_study_paths[var_val] = curr_path

    return all_study_paths


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
            paths.append(('STATE', super_dir / f"{name}_ppo_perfect_memory_memoryless{best_suffix}", 'green'))
        elif info['state_version'] == 'recurrent':
            paths.append(('STATE', super_dir / f"{name}_ppo_perfect_memory{best_suffix}", 'green'))

        curr_path = {
            **info,
            'paths': paths
        }
        all_study_paths[name] = curr_path
    return all_study_paths


def plot_array(all_res: dict, row_mult: int = 10, col_mult: int = 5,
               legend_loc: tuple[float, float] = (0.78, 0.13),
               smoothen_curve: bool = False):
    # first figure out what size array we want
    n_rows, n_cols = (np.array([res['position'] for res in all_res.values()]) + 1).max(axis=0)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(row_mult * n_rows, col_mult * n_cols))

    all_titles_and_colors = []

    for name, info in all_res.items():
        row, col = info['position']
        # assert not ((row == (n_rows - 1)) and (col == (n_cols - 1))), "Legend is blocking a plot!"

        # set up our axis
        if n_rows == 1:
            ax = axes[col]
        else:
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
            if smoothen_curve or info.get('smoothen_curve'):
                mean, conf = smoothen(mean), smoothen(conf)

            x = np.arange(mean.shape[0]) * res['step_multiplier'][0]

            ax.plot(x, mean, label=algo_name, color=colors[color])
            ax.fill_between(x, mean - conf, mean + conf, color=colors[color], alpha=0.35)

            title_color_entry = (algo_name, colors[color])
            if title_color_entry not in all_titles_and_colors:
                all_titles_and_colors.append(title_color_entry)

        if 'range' in info:
            y_min, y_max = info['range']
            ax.set_ylim(y_min, y_max)

    if 'battleship_10' in all_res:
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
    lines = [Line2D([], [], color=color, marker='s', markersize=16, linestyle='None')
             for (_, color) in all_titles_and_colors]
    labels = [label for (label, _) in all_titles_and_colors]
    plt.figlegend(lines, labels, loc=legend_loc)

    fig.supxlabel('Environment steps')
    fig.supylabel(f'Online returns')

    fig.tight_layout()

    plt.show()

    return fig, axes


if __name__ == "__main__":
    to_plot = 'all_envs'

    if to_plot == 'all_envs':
        all_study_settings = generate_study_paths(all_paths)
        row_mult, col_mult = 10, 5
        legend_loc = (0.78, 0.13)
        smoothen_curve = False
    elif to_plot == 'all_masked_mujoco':
        all_study_settings = generate_study_paths(masked_mujoco_paths)
        row_mult, col_mult = 15, 3
        legend_loc = (0.8, 0.0)
        smoothen_curve = True
    elif to_plot == 'ablation_navix_nenvs':
        all_study_settings = generate_ablation_paths(navix_nenvs_paths)
        row_mult, col_mult = 20, 4
        legend_loc = (0.8, 0.3)
        smoothen_curve = False
    elif to_plot == 'ablation_walker_hsize':
        all_study_settings = generate_ablation_paths(walker_hsize_paths)
        row_mult, col_mult = 30, 3
        legend_loc = (0.78, 0.000)
        smoothen_curve = True
    else:
        raise NotImplementedError

    all_res = {}
    for name, info in all_study_settings.items():
        paths = info['paths']
        discounted = info['discounted']
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

    fig, axes = plot_array(all_res, row_mult=row_mult, col_mult=col_mult,
                           legend_loc=legend_loc, smoothen_curve=smoothen_curve)


    save_plot_to = Path(ROOT_DIR, 'results', f'{to_plot}.pdf')

    fig.savefig(save_plot_to, bbox_inches='tight', dpi=100)
    print(f"Saved figure to {save_plot_to}")
