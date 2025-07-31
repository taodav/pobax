from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5]  # Learning rates
lambda0s = [0.1, 0.5, 0.95]
lambda1s = [0.5, 0.7, 0.95]
alphas = [1]
ld_weights = [0.25, 0.5]
vf_coeffs = [0.5]  # Value function coefficients

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [
        {
            'env': 'rocksample_11_11',
            'double_critic': True,
            'action_concat': True,
            'lr': ' '.join(map(str, lrs)),
            'lambda0': lambda0s,
            'lambda1': lambda1s,
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ld_weights,
            'hidden_size': 256,
            'num_envs': 16,
            'entropy_coeff': 0.1,
            'steps_log_freq': 4,
            'update_log_freq': 5,
            'total_steps': int(5e6),
            'seed': 2024,
            'n_seeds': 5,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
