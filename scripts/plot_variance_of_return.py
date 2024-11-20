import matplotlib.pyplot as plt
from pathlib import Path
import orbax.checkpoint
from definitions import ROOT_DIR
from pobax.utils.file_system import load_info
import numpy as np

def plot_returns(study_path):
    num_studies = len(study_path)
    fig, axes = plt.subplots(nrows=1, ncols=num_studies, figsize=(10 * num_studies, 6))

    # Ensure axes is always iterable
    if num_studies == 1:
        axes = [axes]

    all_returns = []

    # Collect all returns to compute common bins
    min_return = np.inf
    max_return = -np.inf

    for _, path in study_path:
        if path.suffix == '.npy':
            restored = load_info(path)
        else:
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            restored = orbax_checkpointer.restore(path)
        return_list = restored['return_list']
        returns = np.array(return_list).squeeze()
        all_returns.append(returns)
        min_return = min(min_return, np.min(returns))
        max_return = max(max_return, np.max(returns))

    # Define common bins for all histograms
    bins = np.linspace(min_return, max_return, 50)

    # Plot each histogram in its own subplot
    for ax, returns, (study, _) in zip(axes, all_returns, study_path):
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Plot histogram of returns
        ax.hist(returns, bins=bins, color='skyblue', edgecolor='black', alpha=0.7)
        ax.set_title(f'Histogram of {study} Discounted Returns')
        ax.set_xlabel('Discounted Return')
        ax.set_ylabel('Frequency')

        # Plot mean and standard deviation lines
        ax.axvline(mean_return, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_return:.2f}')
        ax.axvline(mean_return + std_return, color='green', linestyle='dashed', linewidth=2, label=f'Std Dev: {std_return:.2f}')
        ax.axvline(mean_return - std_return, color='green', linestyle='dashed', linewidth=2)

        ax.legend()

    plt.tight_layout()

    # Save the plot
    parent_dir = Path(ROOT_DIR, 'graphs')
    parent_dir.mkdir(parents=True, exist_ok=True)
    plot_path = parent_dir / 'combined_variance_of_return.png'
    plt.savefig(plot_path)
    print(f"Variance of returns plot saved to {plot_path}")

    plt.show()

if __name__ == "__main__":
    env_name = 'reacher'
    study_path = [
        # ('PPO + RNN_skip', Path(ROOT_DIR, 'results', 'reacher_rnn_skip_ppo_best_variance_of_return.npy')),
        ('PPO + RNN', Path(ROOT_DIR, 'results', 'reacher_rnn_ppo_best_variance_of_return.npy')),
        ('PPO + Memoryless', Path(ROOT_DIR, 'results', 'reacher_memoryless_no_skip_ppo_best_variance_of_return.npy')),
        # ('PPO + Memoryless_skip', Path(ROOT_DIR, 'results', 'reacher_memoryless_ppo_best_variance_of_return.npy')),
    ]

    plot_returns(study_path)

