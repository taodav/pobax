import matplotlib.pyplot as plt
from pathlib import Path
import orbax.checkpoint
from definitions import ROOT_DIR
from pobax.utils.file_system import load_info
import jax.numpy as jnp

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

def plot_epoch_losses(study_path, env_name):
    plt.figure(figsize=(12, 8))

    for study, path, color in study_path:
        if path.suffix == '.npy':
            # print(f"Loading {path}")
            restored = load_info(path)
        else:
            orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
            restored = orbax_checkpointer.restore(path)
        epoch_losses = restored['value_info']['value']
        print(f'{study}:', restored['distance'])

        # Plot each experiment's epoch losses
        plt.plot(jnp.log(epoch_losses), label=study, color=colors[color])

    plt.xlabel('Epoch')
    plt.ylabel('Log Loss')
    plt.title(f'Training Loss for {env_name}')
    plt.legend()
    plt.grid(True)

    # Save the plot
    parent_dir = Path(ROOT_DIR, 'graphs')
    plot_path = parent_dir / f'{env_name}_value_distance_plot.png'
    plt.savefig(plot_path)
    print(f"Epoch loss plot saved to {plot_path}")

    # Display the plot
    plt.show()

if __name__ == "__main__":
    env_name = 'reacher'
    # fname = 'probe.npy'
    study_path =[
        # ('PPO + Memoryless', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_no_skip_ppo', fname), 'yellow'),
        # ('PPO + Memoryless + LD', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_no_skip_ppo_LD', fname), 'cyan'),
        # ('PPO + Memoryless + Skip connection', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_ppo', fname), 'orange'),
        # ('PPO + Memoryless + Skip connection + LD', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_ppo_LD', fname), 'red'),
        # ('PPO + RNN', Path(ROOT_DIR, 'results', f'{env_name}_rnn_ppo', fname), 'dark gray'),
        # ('PPO + RNN + LD', Path(ROOT_DIR, 'results', f'{env_name}_rnn_ppo_LD', fname), 'blue'),
        # ('PPO + RNN Skip', Path(ROOT_DIR, 'results', f'{env_name}_rnn_skip_ppo', fname), 'purple'),
        # ('PPO + RNN Skip + LD', Path(ROOT_DIR, 'results', f'{env_name}_rnn_skip_ppo_LD', fname), 'green'),

        # ('PPO + Memoryless', Path(ROOT_DIR, 'results', f'memoryless_embedding', fname), 'yellow'),
        # ('PPO + Memoryless + LD', Path(ROOT_DIR, 'results', f'memoryless_LD_embedding', fname), 'cyan'),
        # ('PPO + Memoryless + Skip connection', Path(ROOT_DIR, 'results', f'memoryless_skip_embedding', fname), 'orange'),
        # ('PPO + Memoryless + Skip connection + LD', Path(ROOT_DIR, 'results', f'memoryless_skip_LD_embedding', fname), 'red'),
        # ('PPO + RNN', Path(ROOT_DIR, 'results', f'rnn_embedding', fname), 'dark gray'),
        # ('PPO + RNN + LD', Path(ROOT_DIR, 'results', f'rnn_LD_embedding', fname), 'blue'),
        # ('PPO + RNN Skip', Path(ROOT_DIR, 'results', f'rnn_skip_embedding', fname), 'purple'),
        # ('PPO + RNN Skip + LD', Path(ROOT_DIR, 'results', f'rnn_skip_LD_embedding', fname), 'green'),

        ('PPO + Memoryless + TD', Path(ROOT_DIR, 'results', f'distance_mlp_td.npy'), 'yellow'),
        ('PPO + Memoryless + MC', Path(ROOT_DIR, 'results', f'distance_mlp_mc.npy'), 'cyan'),
        ('PPO + RNN_skip + TD', Path(ROOT_DIR, 'results', f'distance_rnn_skip_td.npy'), 'orange'),
        ('PPO + RNN_skip + MC', Path(ROOT_DIR, 'results', f'distance_rnn_skip_mc.npy'), 'red'),
        ('PPO + RNN + TD', Path(ROOT_DIR, 'results', f'distance_rnn_td.npy'), 'dark gray'),
        ('PPO + RNN + MC', Path(ROOT_DIR, 'results', f'distance_rnn_mc.npy'), 'blue'),

        # ('Memoryless PPO + depth 3', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_no_skip_depth3_ppo', fname), 'dark gray'),
        # ('Memoryless PPO + depth 3 + LD', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_no_skip_depth3_ppo_LD', fname), 'blue'),
        # ('Memoryless PPO + depth 5', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_no_skip_depth5_ppo', fname), 'green'),
        # ('Memoryless PPO + depth 5 + LD', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_no_skip_depth5_ppo_LD', fname), 'purple'),
        # ('Memoryless PPO + depth 7', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_no_skip_depth7_ppo', fname), 'yellow'),
        # ('Memoryless PPO + depth 7 + LD', Path(ROOT_DIR, 'results', f'{env_name}_memoryless_no_skip_depth7_ppo_LD', fname), 'cyan'),
    ]

    plot_epoch_losses(study_path, env_name)
