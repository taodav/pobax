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

    for k, (study_name, res, color) in enumerate(all_reses):
        scores = res['scores']
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
            ax.set_title(env_name_to_title.get(env, env))
            if x_upper_lim is not None:
                ax.set_xlim(right=x_upper_lim)
            # ax.margins(x=0.015)
            ax.locator_params(axis='x', nbins=3, min_n_ticks=3)
            ax.locator_params(axis='y', nbins=3)
            ax.spines[['right', 'top']].set_visible(False)

    # Customize legend to use square markers
    legend = plt.legend(loc='lower right')

    # # Change line in legend to square
    # for line in legend.get_lines():
    #     line.set_marker('s')
    #     line.set_markerfacecolor(line.get_color())
    #     line.set_linestyle('')
    #     line.set_markersize(20)  # Increase the marker size

    fig.supxlabel('Environment steps')
    fig.supylabel(f'Online discounted returns ({n_seeds} runs)')
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
    env_names = ['atari', 'cartpole', 'mujoco', 'pong', 'reacher', 'swimmer']
    for env_name in env_names:
        # normal
        study_paths = [
            # ('$\lambda$-discrepancy + PPO', Path(ROOT_DIR, 'results', f'{env_name}_LD_ppo'), 'green'),
            # ('PPO', Path(ROOT_DIR, 'results', f'{env_name}_rnn_ppo'), 'blue'),
            # ('Memoryless PPO', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_ppo'), 'dark gray'),
            # ('PPO', Path(ROOT_DIR, 'results', f'{env_name}_rnn_ppo_best'), 'blue'),
            # ('Memoryless PPO', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_ppo_best'), 'dark gray'),

            # depth

            # ('PPO + RNN', Path(ROOT_DIR, 'results', f'{env_name}_rnn_depth3_ppo_best'), 'blue'),
            # ('Memoryless PPO + depth 3', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_depth3_ppo_best'), 'dark gray'),
            # ('Memoryless PPO + depth 5', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_depth5_ppo_best'), 'red'),
            # ('Memoryless PPO + depth 7', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_depth7_ppo_best'), 'yellow')

            # width
            # ('PPO + RNN', Path(ROOT_DIR, 'results', f'{env_name}_rnn_width128_ppo_best'), 'blue'),
            # ('Memoryless PPO + width 128', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_width128_ppo_best'), 'dark gray'),
            # ('Memoryless PPO + width 256', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_width256_ppo_best'), 'red'),
            # ('Memoryless PPO + width 512', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_width512_ppo_best'), 'yellow')

            # rnn approximator
            # ('PPO + RNN', Path(ROOT_DIR, 'results', f'{env_name}_rnn_ppo_best'), 'blue'),
            # ('MLP approximator', Path(ROOT_DIR, 'results', f'{env_name}_mlp_approximator_ppo_best'), 'dark gray'),
            # ('RNN approximator', Path(ROOT_DIR, 'results', f'{env_name}_rnn_approximator_ppo_best'), 'red'),

            # horizon
            # ('PPO + RNN', Path(ROOT_DIR, 'results', f'{env_name}_rnn_ppo'), 'blue'),
            # ('RNN approximator + Horizon 3', Path(ROOT_DIR, 'results', f'{env_name}_rnn_approximator_horizon3_ppo'), 'green'),
            # ('RNN approximator + Horizon 6', Path(ROOT_DIR, 'results', f'{env_name}_rnn_approximator_horizon6_ppo'), 'red'),
            # ('RNN approximator + Horizon 12', Path(ROOT_DIR, 'results', f'{env_name}_rnn_approximator_horizon12_ppo'), 'yellow'),

            # truncation length
            # ('RNN + Truncation Length 1', Path(ROOT_DIR, 'results', f'{env_name}_rnn_truncation1_ppo_best'), 'blue'),
            # ('RNN + Truncation Length 2', Path(ROOT_DIR, 'results', f'{env_name}_rnn_truncation2_ppo_best'), 'green'),
            # ('RNN + Truncation Length 4', Path(ROOT_DIR, 'results', f'{env_name}_rnn_truncation4_ppo_best'), 'red'),
            # ('RNN + Truncation Length 32', Path(ROOT_DIR, 'results', f'{env_name}_rnn_truncation32_ppo_best'), 'yellow'),
            # ('RNN + Truncation Length 128', Path(ROOT_DIR, 'results', f'{env_name}_rnn_truncation128_ppo_best'), 'purple'),

            # rnn hidden size
            # ('RNN + Hidden Size 4', Path(ROOT_DIR, 'results', f'{env_name}_rnn_hidden4_ppo_best'), 'blue'),
            # ('RNN + Hidden Size 8', Path(ROOT_DIR, 'results', f'{env_name}_rnn_hidden8_ppo_best'), 'green'),
            # ('RNN + Hidden Size 16', Path(ROOT_DIR, 'results', f'{env_name}_rnn_hidden16_ppo_best'), 'red'),
            # ('RNN + Hidden Size 32', Path(ROOT_DIR, 'results', f'{env_name}_rnn_hidden32_ppo_best'), 'yellow'),
            # ('RNN + Hidden Size 64', Path(ROOT_DIR, 'results', f'{env_name}_rnn_hidden64_ppo_best'), 'brown'),
            # ('RNN + Hidden Size 128', Path(ROOT_DIR, 'results', f'{env_name}_rnn_hidden128_ppo_best'), 'purple'),

            # memoryless hidden size
            # ('Memoryless + Hidden Size 4', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_hidden4_ppo_best'), 'blue'),
            # ('Memoryless + Hidden Size 8', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_hidden8_ppo_best'), 'green'),
            # ('Memoryless + Hidden Size 16', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_hidden16_ppo_best'), 'red'),
            # ('Memoryless + Hidden Size 32', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_hidden32_ppo_best'), 'yellow'),
            # ('Memoryless + Hidden Size 64', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_hidden64_ppo_best'), 'brown'),
            # ('Memoryless + Hidden Size 128', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_hidden128_ppo_best'), 'purple'),

            # stack
            # ('PPO + RNN', Path(ROOT_DIR, 'results', f'{env_name}_rnn_ppo_best'), 'blue'),
            # ('Memoryless + Stack 1 + Skip connection', Path(ROOT_DIR, 'saved_results', f'{env_name}_memoryless_stack1_ppo_best'), 'black'),
            # ('RNN skip + Number Repeats 3', Path(ROOT_DIR, 'saved_results', f'{env_name}_rnn_approximator_ppo_best'), 'brown'),
            # ('Memoryless + Stack 1', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_stack1_ppo_best'), 'green'),
            # ('Memoryless + Stack 2', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_stack2_ppo_best'), 'red'),
            # ('Memoryless + Stack 4', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_stack4_ppo_best'), 'yellow'),
            # ('Memoryless + Stack 8', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_stack8_ppo_best'), 'purple'),
            
            # RNN skip hidden
            # ('PPO + RNN', Path(ROOT_DIR, 'results', f'{env_name}_rnn_ppo_best'), 'blue'),
            # ('RNN skip + Hidden 4 + Number repeats 3', Path(ROOT_DIR, 'results', f'{env_name}_rnn_approximator_hidden4_horizon3_ppo_best'), 'blue'),
            # ('RNN skip + Hidden 8 + Number repeats 3', Path(ROOT_DIR, 'results', f'{env_name}_rnn_approximator_hidden8_horizon3_ppo_best'), 'green'),
            # ('RNN skip + Hidden 16 + Number repeats 3', Path(ROOT_DIR, 'results', f'{env_name}_rnn_approximator_hidden16_horizon3_ppo_best'), 'red'),
            # ('RNN skip + Hidden 32 + Number repeats 3', Path(ROOT_DIR, 'results', f'{env_name}_rnn_approximator_hidden32_horizon3_ppo_best'), 'yellow'),
            # ('RNN skip + Hidden 64 + Number repeats 3', Path(ROOT_DIR, 'results', f'{env_name}_rnn_approximator_hidden64_horizon3_ppo_best'), 'brown'),
            # ('RNN skip + Hidden 128 + Number repeats 3', Path(ROOT_DIR, 'results', f'{env_name}_rnn_approximator_hidden128_horizon3_ppo_best'), 'purple'),

            # Observation + Memoryless
            # ('PPO + RNN', Path(ROOT_DIR, 'results', f'{env_name}_rnn_ppo'), 'blue'),
            # ('Memoryless + Stack Observation 1', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_observation1_ppo'), 'dark gray'),
            # ('Memoryless + Stack Observation 2', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_observation2_ppo'), 'green'),
            # ('Memoryless + Stack Observation 4', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_observation4_ppo'), 'red'),
            # ('Memoryless + Stack Observation 8', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_observation8_ppo'), 'yellow'),

            # Observation + RNN skip
            ('PPO + RNN', Path(ROOT_DIR, 'results', f'{env_name}_rnn_ppo'), 'blue'),
            ('RNN skip + Stack Observation 1', Path(ROOT_DIR, 'results', f'{env_name}_rnn_skip_observation1_ppo'), 'dark gray'),
            ('RNN skip + Stack Observation 2', Path(ROOT_DIR, 'results', f'{env_name}_rnn_skip_observation2_ppo'), 'green'),
            ('RNN skip + Stack Observation 4', Path(ROOT_DIR, 'results', f'{env_name}_rnn_skip_observation4_ppo'), 'red'),
            ('RNN skip + Stack Observation 8', Path(ROOT_DIR, 'results', f'{env_name}_rnn_skip_observation8_ppo'), 'yellow'),


            # reacher
            # ('PPO + RNN', Path(ROOT_DIR, 'results', f'{env_name}_rnn_ppo'), 'blue'),
            # ('RNN skip', Path(ROOT_DIR, 'results', f'{env_name}_rnn_approximator_ppo'), 'red'),
            # ('Memoryless PPO', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_ppo'), 'dark gray'),
        ]

        # fixedlambda
        # study_paths = [
        #     ('$\lambda$-discrepancy + PPO', Path(ROOT_DIR, 'results', f'{env_name}_fixedlambda_LD_ppo'), 'green'),
        #     ('PPO', Path(ROOT_DIR, 'results', f'{env_name}_fixedlambda_ppo'), 'blue'),
        # ]

        # best
        # study_paths = [
        #     ('PPO + RNN + LD', Path(ROOT_DIR, 'results', f'{env_name}_LD_ppo_best'), 'green'),
        #     ('PPO + RNN', Path(ROOT_DIR, 'results', f'{env_name}_ppo_best'), 'blue'),
        #     # ('Memoryless PPO', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_ppo_best'), 'dark gray'),
        # ]

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

            if 'all_hyperparams' in best_res:
                step_multiplier = best_res['all_hyperparams']['total_steps'] // best_res['scores'].shape[0]
            else:
                hyperparams_dir = study_path.parent.parent / 'scripts' / 'hyperparams'
                study_hparam_filename = study_path.stem + '.py'
                hyperparam_path = find_file_in_dir(study_hparam_filename, hyperparams_dir)
                print(f"Hyperparam path: {hyperparam_path}")
                step_multiplier = get_total_steps_multiplier(best_res['scores'].shape[0], hyperparam_path)
            best_res['step_multiplier'] = [step_multiplier] * len(best_res['envs'])

        fig, axes = plot_reses(all_reses, individual_runs=False, n_rows=2)

        save_plot_to = Path(ROOT_DIR, 'graphs', f'{plot_name}_observation_rnn_skip.jpg')

        fig.savefig(save_plot_to, bbox_inches='tight')
        print(f"Saved figure to {save_plot_to}")