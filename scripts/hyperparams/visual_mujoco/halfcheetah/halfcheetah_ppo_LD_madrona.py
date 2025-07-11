from pathlib import Path

# Set up the experiment name based on the script filename
file = __file__  # Use this in actual script to dynamically get the file name
exp_name = Path(file).stem

# Define various hyperparameter values
lrs = [2.5e-4, 2.5e-5]  # Learning rates
lambda0s = [0.1, 0.95]
lambda1s = [0.5, 0.7, 0.95]
alphas = [1]
ld_weights = [0.25, 0.5]
vf_coeffs = [0.5]  # Value function coefficients
entropy_coeffs = [0.01]  # Entropy coefficients

# Prepare the hyperparameters dictionary
hparams = {
    'file_name': f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [{
        'env': 'halfcheetah_pixels',
        'double_critic': True,
        'action_concat': True,
        'lr': lrs,
        'lambda0': lambda0s,
        'lambda1': lambda1s,
        'alpha': ' '.join(map(str, alphas)),
        'ld_weight': ld_weights,
        'hidden_size': 512,
        'entropy_coeff': 0.01,
        'steps_log_freq': 8,
        'update_log_freq': 10,
        'total_steps': int(5e6),
        'seed': [2024 + i for i in range(3)],
        'n_seeds': 1,
        'platform': 'gpu',
        'debug': True,
        'study_name': exp_name
    }]
}
