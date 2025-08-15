import importlib
import pickle
from pathlib import Path
from typing import Tuple

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
}

def plot_reses(all_reses: list[tuple], n_rows: int = 2,
               individual_runs: bool = False,
               plot_title: str = None,
               discounted: bool = False,
               ylims: Tuple[float, float] = None):
    plt.rcParams.update({'font.size': 32})

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
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 10))

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
                mean[k] = mean_over_steps.mean(axis=-1)
                std_err[k] = sem(mean_over_steps, axis=-1)
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
                # -2 index is seeds.
                for j in range(mean_over_steps.shape[-2]):
                    alpha = 1 / mean_over_steps.shape[-2]
                    m = mean_over_steps[..., j, env_idx] if isinstance(mean_over_steps, np.ndarray) else mean_over_steps[env_idx][..., j]
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
    legend = plt.legend(loc='upper left')

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

    ylims = None

    # discounted = False
    # env_name = 'battleship_10'
    # super_dir = 'battleship'
    # best = False

    #discounted = False
    #env_name = 'rocksample_11_11'
    #super_dir = 'rocksample_11_11'
    #best = False
    # ylims = (0, 40)  # for rocksample_11_11 hsize

    discounted = False
    env_name = 'navix_01'
    super_dir = 'navix_01'
    best = True
    ylims = (0, 1)  # for navix

    # discounted = True
    # env_name = 'walker_v'
    # super_dir = 'walker_v'
    # best = False

    best_str = '_best' if best else ''
    super_dir += best_str


    plot_name = f'{env_name}_{hyperparam_type}'

    # normal
    study_paths = [

        # ('Trace', Path(ROOT_DIR, 'results', 'trace_experiments', f'{env_name}_ppo_trace_memoryless'), 'blue'),
        # ('SF', Path(ROOT_DIR, 'results', 'sf_ppo_rr', f'{env_name}_sf_ppo_rr'), 'cyan'),
        # ('SF discrep', Path(ROOT_DIR, 'results', 'sf_ppo_rr', f'{env_name}_sf_ppo_rr_discrep'), 'yellow'),

        # ('SF', Path(ROOT_DIR, 'results', 'gd_sf_grid_sweep_rew_concat', f'{env_name}_ppo_gd_sf_grid_sweep'), 'cyan'),
        # ('SF discrep', Path(ROOT_DIR, 'results', 'gd_sf_grid_sweep_rew_concat', f'{env_name}_ppo_gd_sf_grid_sweep_discrep'), 'yellow'),

        # ('SF raw hs', Path(ROOT_DIR, 'results', 'gd_sf_hs', f'{env_name}_ppo_gd_sf_hs'), 'cyan'),
        # ('SF raw hs discrep', Path(ROOT_DIR, 'results', 'gd_sf_hs', f'{env_name}_ppo_gd_sf_hs_discrep'), 'yellow'),
        # ('SF raw hs diff', Path(ROOT_DIR, 'results', 'gd_sf_hs', env_name, f'{env_name}_ppo_gd_sf_hs_diff'), 'cyan'),
        # ('SF raw hs diff discrep', Path(ROOT_DIR, 'results', 'gd_sf_hs', env_name, f'{env_name}_ppo_gd_sf_hs_diff_discrep'), 'yellow'),

        # ('SF random proj hs', Path(ROOT_DIR, 'results', 'gd_sf_random_proj_hs', f'{env_name}_ppo_gd_sf_random_proj_hs'), 'cyan'),
        # ('SF random proj hs discrep', Path(ROOT_DIR, 'results', 'gd_sf_random_proj_hs', f'{env_name}_ppo_gd_sf_random_proj_hs_discrep'), 'yellow'),
        # ('SF random proj hs diff', Path(ROOT_DIR, 'results', 'gd_sf_random_proj_hs', env_name, f'{env_name}_ppo_gd_sf_random_proj_hs_diff'), 'cyan'),
        # ('SF random proj hs diff discrep', Path(ROOT_DIR, 'results', 'gd_sf_random_proj_hs', env_name, f'{env_name}_ppo_gd_sf_random_proj_hs_diff_discrep'), 'yellow'),

        # ('SF random proj obs', Path(ROOT_DIR, 'results', 'gd_sf_random_proj_obs', f'{env_name}_ppo_gd_sf_random_proj_obs'), 'cyan'),
        # ('SF random proj obs discrep', Path(ROOT_DIR, 'results', 'gd_sf_random_proj_obs', f'{env_name}_ppo_gd_sf_random_proj_obs_discrep'), 'yellow'),
        # ('SF random proj obs diff', Path(ROOT_DIR, 'results', 'gd_sf_random_proj_obs', env_name, f'{env_name}_ppo_gd_sf_random_proj_obs_diff'), 'cyan'),
        # ('SF random proj obs diff discrep', Path(ROOT_DIR, 'results', 'gd_sf_random_proj_obs', env_name, f'{env_name}_ppo_gd_sf_random_proj_obs_diff_discrep'), 'yellow'),

        # ('SF obs', Path(ROOT_DIR, 'results', f'{env_name}_ppo_gd_sf_obs'), 'blue'),
        # ('SF random proj. obs', Path(ROOT_DIR, 'results', 'gd_sf_hs', f'{env_name}_ppo_gd_sf_hs'), 'cyan'),
        # ('SF random proj. obs discrep', Path(ROOT_DIR, 'results', 'gd_sf_hs', f'{env_name}_ppo_gd_sf_hs_discrep'), 'cyan'),
        # ('SF obs discrep', Path(ROOT_DIR, 'results', 'gd_sf_obs_discrep', f'{env_name}_ppo_gd_sf_obs_discrep'), 'cyan'),
        # ('SF obs diff', Path(ROOT_DIR, 'results', 'gd_sf_obs_diff', env_name, f'{env_name}_ppo_gd_sf_obs_diff'), 'cyan'),
        # ('SF obs diff discrep', Path(ROOT_DIR, 'results', 'gd_sf_obs_diff', env_name, f'{env_name}_ppo_gd_sf_obs_diff_discrep'), 'yellow'),

        # ('SF encoded obs', Path(ROOT_DIR, 'results', 'gd_sf_enc_obs', f'{env_name}_ppo_gd_sf_enc_obs'), 'cyan'),
        # ('SF encoded obs discrep', Path(ROOT_DIR, 'results', 'gd_sf_enc_obs', f'{env_name}_ppo_gd_sf_enc_obs_discrep'), 'yellow'),

        # ('RNN', Path(ROOT_DIR, 'results', 'qr_ppo', super_dir, f'{env_name}_ppo_qr'), 'purple'),
        # ('LD', Path(ROOT_DIR, 'results', 'qr_ppo', super_dir, f'{env_name}_ppo_qr_discrep'), 'blue'),
        # ('Ent', Path(ROOT_DIR, 'results', 'qr_ppo', super_dir, f'{env_name}_ppo_qr_ent'), 'yellow'),
        ('STATE', Path(ROOT_DIR, 'results','navix_01_ppo_gd_sf_obs_diff_discrep_F'), 'green'),
        ('GD', Path(ROOT_DIR, 'results','navix_01_ppo_gd_sf_obs_diff_discrep'), 'yellow'),
        # ('Memoryless', Path(ROOT_DIR, 'results', 'qr_ppo', env_name, f'{env_name}_ppo_qr_memoryless{best_str}'), 'dark gray'),
        # ('STATE', Path(ROOT_DIR, 'results', 'qr_ppo', env_name, f'{env_name}_ppo_qr_perfect_memory_memoryless{best_str}'), 'green'),

        # ('RNN', Path(ROOT_DIR, 'results', super_dir, f'{env_name}_ppo{best_str}'), 'purple'),
        # ('LD', Path(ROOT_DIR, 'results', super_dir, f'{env_name}_ppo_LD{best_str}'), 'blue'),
        # ('Memoryless', Path(ROOT_DIR, 'results', super_dir, f'{env_name}_ppo_memoryless{best_str}'), 'dark gray'),
        # ('STATE', Path(ROOT_DIR, 'results', super_dir, f'{env_name}_ppo_perfect_memory{best_str}'), 'green'),
        # ('TRANFORMER', Path(ROOT_DIR, 'results', super_dir, f'{env_name}_transformer'), 'cyan'),
    ]

    # env_name = 'rocksample_11_11'
    # sweep_var = 'nenvs'
    # # nenvs = 128
    # # ylims = (0., 0.68)  # for navix_01 nenvs
    # # ylims = (50, 2500)  # for walker_v hsize
    # ylims = (0, 40)  # for rocksample_11_11 hsize
    # # ylims = None
    #
    # discounted = False
    # plot_discrep = False
    #
    # plot_discrep_str = '_discrep' if plot_discrep else ''
    # plot_name = f'{env_name}_{hyperparam_type}_{sweep_var}{plot_discrep_str}'
    #
    # study_paths = [
    #     # ('0 random rewards', Path(ROOT_DIR, 'results', 'sf_ppo_rr', f'{env_name}_sf_ppo_rr{plot_discrep_str}', f'rocksample_11_11_sf_ppo_rr{plot_discrep_str}_num_rr_0'), 'cyan'),
    #     # ('2 random rewards', Path(ROOT_DIR, 'results', 'sf_ppo_rr', f'{env_name}_sf_ppo_rr{plot_discrep_str}', f'rocksample_11_11_sf_ppo_rr{plot_discrep_str}_num_rr_2'), 'yellow'),
    #     # ('8 random rewards', Path(ROOT_DIR, 'results', 'sf_ppo_rr', f'{env_name}_sf_ppo_rr{plot_discrep_str}', f'rocksample_11_11_sf_ppo_rr{plot_discrep_str}_num_rr_8'), 'orange'),
    #
    #     ('8 envs', Path(ROOT_DIR, 'results', 'qr_nenvs', f'rocksample_11_11_ppo_qr{plot_discrep_str}_{sweep_var}_8'), 'cyan'),
    #     ('16 envs', Path(ROOT_DIR, 'results', 'qr_nenvs', f'rocksample_11_11_ppo_qr{plot_discrep_str}_{sweep_var}_16'), 'cyan'),
    #     ('32 envs', Path(ROOT_DIR, 'results', 'qr_nenvs', f'rocksample_11_11_ppo_qr{plot_discrep_str}_{sweep_var}_32'), 'yellow'),
    #     ('64 envs', Path(ROOT_DIR, 'results', 'qr_nenvs', f'rocksample_11_11_ppo_qr{plot_discrep_str}_{sweep_var}_64'), 'orange'),
    #
    #     # ('SF discrep', Path(ROOT_DIR, 'results', 'sf_ppo_rr', f'{env_name}_sf_ppo_rr_discrep'), 'yellow'),
    #     # ('RNN', Path(ROOT_DIR, 'results', f'{env_name}_{sweep_var}_sweep/{env_name}_ppo_{sweep_var}_sweep', f'{env_name}_ppo_{sweep_var}_sweep_{sweep_var}_{nenvs}'), 'purple'),
    #     # ('RNN', Path(ROOT_DIR, 'results', f'entropy_sweep_{env_name}'), 'purple'),
    #     # ('RNN + Two Heads', Path(ROOT_DIR, 'results', f'second_head_sweep_{env_name}'), 'cyan'),
    #     # ('RNN + LD', Path(ROOT_DIR, 'results', super_dir, f'{env_name}_ppo_LD'), 'blue'),
    #     # ('RNN + LD exploration', Path(ROOT_DIR, 'results', f'ld_exploration_{env_name}'), 'blue'),
    #     # ('Memoryless', Path(ROOT_DIR, 'results', f'{env_name}_{sweep_var}_sweep/{env_name}_ppo_memoryless_{sweep_var}_sweep', f'{env_name}_ppo_memoryless_{sweep_var}_sweep_{sweep_var}_{nenvs}'), 'dark gray'),
    #     # ('STATE', Path(ROOT_DIR, 'results', f'{env_name}_{sweep_var}_sweep/{env_name}_ppo_perfect_memory_{sweep_var}_sweep', f'{env_name}_ppo_perfect_memory_{sweep_var}_sweep_{sweep_var}_{nenvs}'), 'green'),
    #     # ('TRANFORMER', Path(ROOT_DIR, 'results', super_dir, f'{env_name}_transformer'), 'cyan'),
    # ]

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
                    for key in ['hyperparams', 'scores']:
                        if env in best_res[key]:
                            best_res[key][fully_observable_to_base[env]] = best_res[key][env]
                            del best_res[key][env]
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

    fig, axes = plot_reses(all_reses, individual_runs=False, n_rows=3, plot_title=plot_name,
                           discounted=discounted, ylims=ylims)

    discount_str = '_discounted' if discounted else ''
    save_plot_to = Path(ROOT_DIR, 'results', f'{plot_name}{discount_str}.pdf')

    fig.savefig(save_plot_to, bbox_inches='tight')
    print(f"Saved figure to {save_plot_to}")

