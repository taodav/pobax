from pathlib import Path

# Set up the experiment name based on the script filename
file = __file__  # Use this in actual script to dynamically get the file name
exp_name = Path(file).stem

# Prepare the hyperparameters dictionary
hparams = {
    'file_name': f'runs_{exp_name}.txt',
    'entry': 'pobax.algos.ppo',
    'args': [{
        'env': 'Breakout-MinAtar',
        'memoryless': True,
        'approximator': 'mlp',
        'lr': '0.001',
        'depth': 5,
        'lambda0': '0.95',
        'vf_coeff': '0.5',
        'entropy_coeff': '0.01',
        'clip_eps': 0.2,
        'max_grad_norm': 0.5,
        'hidden_size': 4,
        'num_minibatches': 4,
        'num_envs': 4,
        'num_steps': 128,
        'steps_log_freq': 8,
        'update_log_freq': 10,
        'update_epochs': 4,
        'total_steps': int(5e6),
        'seed': 2024,
        'n_seeds': 30,
        'platform': 'gpu',
        'debug': False,
        'study_name': exp_name
    },
    {
        'env': 'SpaceInvaders-MinAtar',
        'memoryless': True,
        'approximator': 'mlp',
        'lr': '0.001',
        'depth': 5,
        'lambda0': '0.95',
        'vf_coeff': '0.5',
        'entropy_coeff': '0.01',
        'clip_eps': 0.2,
        'max_grad_norm': 0.5,
        'hidden_size': 4,
        'num_minibatches': 4,
        'num_envs': 4,
        'num_steps': 128,
        'steps_log_freq': 8,
        'update_log_freq': 10,
        'update_epochs': 4,
        'total_steps': int(5e6),
        'seed': 2024,
        'n_seeds': 30,
        'platform': 'gpu',
        'debug': False,
        'study_name': exp_name
    }]
}