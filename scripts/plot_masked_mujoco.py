from pathlib import Path
import pickle

import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt

from pobax.utils.plot import mean_confidence_interval

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
    'dark green': '#27ae60',
    'cyan': '#48dbe5',
    'blue': '#3180df',
    'purple': '#9d79cf',
    'brown': '#886a2c',
    'white': '#ffffff',
    'light gray': '#d5d5d5',
    'dark gray': '#666666',
    'black': '#000000'
}

env_info = {
    # 'CartPole-P-v0': {'F': 'CartPole-F-v0'},
    # 'CartPole-V-v0': {'F': 'CartPole-F-v0'},
    # 'Pendulum-P-v0': {'F': 'Pendulum-F-v0'},
    # 'Pendulum-V-v0': {'F': 'Pendulum-F-v0'},
    'Hopper-P-v0': {'F': 'Hopper-F-v0'},
    'Hopper-V-v0': {'F': 'Hopper-F-v0'},
    'Walker-P-v0': {'F': 'Walker-F-v0'},
    'Walker-V-v0': {'F': 'Walker-F-v0'},
    'Ant-P-v0': {'F': 'Ant-F-v0'},
    'Ant-V-v0': {'F': 'Ant-F-v0'},
    'HalfCheetah-P-v0': {'F': 'HalfCheetah-F-v0'},
    'HalfCheetah-V-v0': {'F': 'HalfCheetah-F-v0'},
}

def calc_means_and_confidences(all_reses: list[tuple],
                               all_F_reses: list[tuple]) -> dict:
    envs = [env for env in all_reses[0][1]['envs'] if env in env_info]
    final_res = {}
    for env in sorted(envs):
        if env not in final_res:
            final_res[env] = {}

        def calc_m_and_c(res, env_name):
            dim_ref = res['dim_ref'][:]

            # get environment scores
            env_idx = res['envs'].index(env_name)
            dim_to_index = dim_ref.index('env')
            scores = res['scores'].take(env_idx, axis=dim_to_index)
            dim_ref.remove('env')

            # average over num_steps
            steps_dim = dim_ref.index('num_steps')
            mean_over_steps = scores.mean(axis=steps_dim)
            dim_ref.remove('num_steps')

            # Finally, calc mean + confidence interval over seeds
            seed_dim = dim_ref.index('seeds')
            mean, conf = mean_confidence_interval(mean_over_steps, axis=seed_dim)
            x_axis_multiplier = res['step_multiplier'][env_idx]
            x = np.arange(mean.shape[0]) * x_axis_multiplier
            parsed_res = {
                'x': x,
                'mean': mean,
                'confidence': conf,
                'seeds': mean_over_steps.shape[seed_dim],
            }
            return parsed_res

        # parse all_reses
        for k, (study_name, res, color) in enumerate(all_reses):
            final_res[env][study_name] = (calc_m_and_c(res, env), color)

        # parse all_F_reses
        for k, (study_name, res, color) in enumerate(all_F_reses):
            final_res[env][study_name] = (calc_m_and_c(res, env_info[env]['F']), color)
    return final_res


def plot_res(means_and_confidences: dict,
             n_rows: int = 2,
             discounted: bool = False):
    plt.rcParams.update({'font.size': 32})

    envs = list(sorted(means_and_confidences.keys()))

    n_rows = min(n_rows, len(envs))
    n_cols = max((len(envs) + 1) // n_rows, 1) if len(envs) > 1 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 12))

    for k, env in enumerate(envs):
        row = k // n_cols - 1
        col = k % n_cols

        if len(envs) == 1:
            ax = axes
        else:
            ax = axes[row, col] if n_cols > 1 else axes[k]
        for label, (res, color) in means_and_confidences[env].items():
            # x_upper_lim = env_name_to_x_upper_lim.get(env, None)
            x, mean, conf = res['x'], res['mean'], res['confidence']

            ax.plot(x, mean, label=label, color=colors[color])
            ax.fill_between(x, mean - conf, mean + conf,
                            color=colors[color], alpha=0.35)
        ax.set_title(env)
        # if x_upper_lim is not None:
        #     ax.set_xlim(right=x_upper_lim)
        # ax.margins(x=0.015)
        ax.locator_params(axis='x', nbins=3, min_n_ticks=3)
        ax.locator_params(axis='y', nbins=3)
        ax.spines[['right', 'top']].set_visible(False)

    # Customize legend to use square markers
    legend = plt.legend(loc='center right')

    # # Change line in legend to square
    # for line in legend.get_lines():
    #     line.set_marker('s')
    #     line.set_markerfacecolor(line.get_color())
    #     line.set_linestyle('')
    #     line.set_markersize(20)  # Increase the marker size

    fig.supxlabel('Environment steps')
    if discounted:
        fig.supylabel(f'Online discounted returns ({res["seeds"]} runs)')
    else:
        fig.supylabel(f'Online returns ({res["seeds"]} runs)')

    fig.tight_layout()

    plt.show()
    return fig, axes

if __name__ == "__main__":
    discounted = True
    super_dir = 'masked_mujoco'
    env_name = 'masked_mujoco'


    study_paths = [
        ('RNN', Path(ROOT_DIR, 'results', super_dir, f'{env_name}_ppo'), 'purple'),
        # ('RNN + LD', Path(ROOT_DIR, 'results', super_dir, f'{env_name}_ppo_LD'), 'blue'),
        ('OBSERVATION', Path(ROOT_DIR, 'results', super_dir, f'{env_name}_ppo_memoryless'), 'dark gray'),
        ('FULL STATE', Path(ROOT_DIR, 'results', super_dir, f'{env_name}_ppo_perfect_memory_memoryless'), 'green'),
    ]
    plot_name = study_paths[0][1].stem

    all_reses = []
    all_F_reses = []

    for name, study_path, color in study_paths:
        fname = "best_hyperparam_per_env_res.pkl"
        if discounted:
            fname = "best_hyperparam_per_env_res_discounted.pkl"

        with open(study_path / fname, "rb") as f:
            best_res = pickle.load(f)


        step_multiplier = best_res['all_hyperparams']['total_steps'] // best_res['scores'].shape[0]
        best_res['step_multiplier'] = [step_multiplier] * len(best_res['envs'])

        if 'STATE' in name:
            all_F_reses.append((name, best_res, color))
        else:
            all_reses.append((name, best_res, color))

    means_and_confidences = calc_means_and_confidences(all_reses, all_F_reses)
    fig, axes = plot_res(means_and_confidences, n_rows=2,
                         discounted=discounted)

    discount_str = '_discounted' if discounted else ''
    save_plot_to = Path(ROOT_DIR, 'results', f'{plot_name}{discount_str}.pdf')

    fig.savefig(save_plot_to, bbox_inches='tight')
    print(f"Saved figure to {save_plot_to}")
