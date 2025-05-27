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

def plot_rnd_loss(path, rnd_loss):
    """
    Plots the RND loss curves over time and saves the plot as "rnd_loss.pdf" in the specified path.
    
    Parameters:
        path (str or Path): Directory where the plot will be saved.
        rnd_loss (array-like): A loss array with shape (1,1,1,1,1,1,1,5,1953),
                               where the 5 represents the number of seeds.
                               
    The function squeezes the input array to remove singleton dimensions,
    computes the mean and standard deviation across the seed dimension,
    and then plots the mean loss over time with an error band representing one standard deviation.
    """
    # Remove singleton dimensions. Expected shape becomes (num_seeds, time_steps) i.e. (5, 1953).
    print(f"Original shape of rnd_loss: {rnd_loss.shape}")
    loss = np.squeeze(rnd_loss)
    # loss = np.expand_dims(loss, axis=0)
    print(f"Shape of loss after squeezing: {loss.shape}")
    if loss.ndim != 2:
        raise ValueError("After squeezing, expected a 2D array of shape (num_seeds, time_steps)")
    
    num_seeds, time_steps = loss.shape
    
    # Compute the mean and standard deviation over the seeds dimension.
    mean_loss = np.mean(loss, axis=0)
    print('Mean loss', mean_loss)
    std_loss = np.std(loss, axis=0)
    
    plt.figure(figsize=(10, 6))
    time_axis = np.arange(time_steps)
    
    # Plot the mean loss curve.
    plt.plot(time_axis, mean_loss, label="Mean Loss", color="blue", linewidth=2)
    
    # Plot the error band (mean ± standard deviation).
    plt.fill_between(time_axis, mean_loss - std_loss, mean_loss + std_loss, 
                     color="blue", alpha=0.3, label="Std Dev")
    
    plt.xlabel("Update Step")
    plt.ylabel("Loss")
    plt.title("RND Loss over Time")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    
    # Save the plot to the specified path as "rnd_loss.pdf"
    path = Path(path)
    save_path = path / "rnd_loss.pdf"
    plt.savefig(save_path)
    plt.close()

def plot_reses(all_reses: list[tuple], env_name, n_rows: int = 2,
               individual_runs: bool = False):
    plt.rcParams.update({'font.size': 32})

    # check to see that all our envs are the same across all reses.
    for _, x, _ in all_reses:
        for i in range(len(x['envs'])):
            if x['envs'][i].endswith('pixels'):
                x['envs'][i] = x['envs'][i][:-7]
                print(x['envs'][i])
            if x['envs'][i].startswith('Navix-DMLab'):
                x['envs'][i] = env_name
                print(x['envs'][i])
    all_envs = [set(x['envs']) for _, x, _ in all_reses]
    for envs in all_envs:
        assert envs == all_envs[0]

    envs = list(sorted(all_envs[0]))

    n_rows = min(n_rows, len(envs))
    n_cols = max((len(envs) + 1) // n_rows, 1) if len(envs) > 1 else 1
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(30, 20))

    for k, (study_name, res, color) in enumerate(all_reses):
        scores = res['scores']
        if 'rnd_loss' in res:
            rnd_loss = res['rnd_loss']
            # Define an output folder for rnd_loss plots.
            rnd_output_folder = Path(ROOT_DIR, 'results')
            plot_rnd_loss(rnd_output_folder, rnd_loss)
            print(f"Saved RND loss plot for {study_name} in {rnd_output_folder}")
        print(scores.shape)
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
    env_name = 'navix_02'

    # normal
    study_paths = [
        # ('$\lambda$-discrepancy + Quantile PPO', Path(ROOT_DIR, 'results', f'{env_name}_quantile_LD_ppo'), 'green'),
        ('PPO + Memoryless', Path(ROOT_DIR, 'results', f'{env_name}_ppo_memoryless'), 'dark gray'),
        ('PPO + RNN', Path(ROOT_DIR, 'results', f'{env_name}_ppo'), 'purple'),
        ('PPO + LD', Path(ROOT_DIR, 'results', f'{env_name}_ppo_LD'), 'blue'),
        # ('PPO + TRANSFORMER + No Frame', Path(ROOT_DIR, 'results', f'{env_name}_transformer'), 'yellow'),
        # ('PPO + TRANSFORMER', Path(ROOT_DIR, 'results', f'{env_name}_transformer_best'), 'cyan'),
        # ('PPO + RND MEMORY', Path(ROOT_DIR, 'results', f'{env_name}_ppo_memory_rnd'), 'blue'),
        # ('PPO + RND TRACE + Lambda 0', Path(ROOT_DIR, 'results', f'{env_name}_ppo_rnd_trace_lambda0'), 'cyan'),
        # ('PPO + RND TRACE + Lambdas', Path(ROOT_DIR, 'results', f'{env_name}_ppo_rnd_trace'), 'green'),
        # ('PPO + RND TRACE + Trace Obs', Path(ROOT_DIR, 'results', f'{env_name}_ppo_rnd_trace_in_obs'), 'yellow'),
        # ('PPO + MEMORYLESS Trace', Path(ROOT_DIR, 'results', f'{env_name}_ppo_memoryless_trace'), 'orange'),
        # ('PPO + TRACE RNN', Path(ROOT_DIR, 'results', f'{env_name}_ppo_rnd_trace_rnn_in_obs'), 'dark gray'),
        ('PPO + OBSERVABLE', Path(ROOT_DIR, 'results', f'{env_name}_ppo_observable'), 'green'),
        # ('TEST', Path(ROOT_DIR, 'results', f'test_trace_rnd'), 'blue'),
        # ('Perfect Memory PPO (NN)', Path(ROOT_DIR, 'results', f'{env_name}_perfect_memory_memoryless_ppo'), 'pink'),
        # ('PPO (RNN)', Path(ROOT_DIR, 'results', f'{env_name}_ppo'), 'blue'),
        # ('PPO (NN)', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_ppo'), 'dark gray'),
    ]

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
            fname = "best_hyperparam_per_env_res_undiscounted.pkl"

        with open(study_path / fname, "rb") as f:
            best_res = pickle.load(f)

        all_reses.append((name, best_res, color))

        if 'all_hyperparams' in best_res:
            step_multiplier = best_res['all_hyperparams']['total_steps'] // best_res['scores'].shape[0]
        else:
            print(study_path)
            hyperparams_dir = study_path.parent.parent / 'scripts' / 'hyperparams'
            study_hparam_filename = study_path.stem + '.py'
            hyperparam_path = find_file_in_dir(study_hparam_filename, hyperparams_dir)
            step_multiplier = get_total_steps_multiplier(best_res['scores'].shape[0], hyperparam_path)
        best_res['step_multiplier'] = [step_multiplier] * len(best_res['envs'])

    fig, axes = plot_reses(all_reses, env_name, individual_runs=False, n_rows=3)

    save_plot_to = Path(ROOT_DIR, 'results', f'{plot_name}_undiscounted.pdf')

    fig.savefig(save_plot_to, bbox_inches='tight')
    print(f"Saved figure to {save_plot_to}")
