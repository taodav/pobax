from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-4]  # Learning rates
lambda0s = [0.5]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0.25]
vf_coeffs = [0.5]  # Value function coefficients

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [
        {
            'env': 'pocman',
            'double_critic': True,
            'action_concat': True,
            'lr': lrs,
            'lambda0': ' '.join(map(str, lambda0s)),
            'lambda1': lambda1s,
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ld_weights,
            'hidden_size': 512,
            'num_envs': 16,
            'entropy_coeff': 0.05,
            'steps_log_freq': 8,
            'update_log_freq': 10,
            'total_steps': int(1e7),
            'seed': 2026,
            'n_seeds': 30,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
