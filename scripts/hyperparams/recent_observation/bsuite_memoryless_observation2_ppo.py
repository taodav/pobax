from pathlib import Path

# Set up the experiment name based on the script filename
file = __file__  # Use this in actual script to dynamically get the file name
exp_name = Path(file).stem

# Define various hyperparameter values
lrs = [2.5e-3, 2.5e-4, 2.5e-5]  # Learning rates
lambda0s = [0.1, 0.3, 0.5, 0.7, 0.95]  # GAE lambda values
vf_coeffs = [0.5]  # Value function coefficients
entropy_coeffs = [0.01]  # Entropy coefficients

# Prepare the hyperparameters dictionary
hparams = {
    'file_name': f'runs_{exp_name}.txt',
    'entry': 'pobax.algos.ppo',
    'args': [{
        'env': ['Catch-bsuite', 'DeepSea-bsuite', 'MemoryChain-bsuite', 'UmbrellaChain-bsuite', 'MNISTBandit-bsuite'],
        'memoryless': True,
        'approximator': 'mlp',
        'depth': 3,
        'num_stack': 1,
        'num_observation': 2,
        'lr': ' '.join(map(str, lrs)),
        'lambda0': ' '.join(map(str, lambda0s)),
        'vf_coeff': ' '.join(map(str, vf_coeffs)),
        'entropy_coeff': ' '.join(map(str, entropy_coeffs)),
        'clip_eps': 0.2,
        'max_grad_norm': 0.5,
        'hidden_size': 128,
        'num_minibatches': 4,
        'num_envs': 4,
        'num_steps': 128,
        'steps_log_freq': 4,
        'update_log_freq': 5,
        'update_epochs': 4,
        'total_steps': int(1e6),
        'seed': 2020,
        'n_seeds': 5,
        'platform': 'gpu',
        'debug': False,
        'study_name': exp_name
    }]
}