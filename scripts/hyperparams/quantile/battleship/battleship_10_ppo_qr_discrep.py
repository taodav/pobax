from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5, 2.5e-6]
lambda0s = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
lambda1s = [0.95]
quantile_entropy_coeffs = [0.]
ld_weights = [0., 0.25, 0.5]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.qr_ppo',
    'args': [
        {
            'env': 'battleship_10',
            'double_critic': True,
            'action_concat': True,
            'lr': ' '.join(map(str, lrs)),
            'lambda0': ' '.join(map(str, lambda0s)),
            'lambda1': ' '.join(map(str, lambda1s)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'quantile_entropy_coeff': ' '.join(map(str, quantile_entropy_coeffs)),
            'hidden_size': 512,
            'num_envs': 32,
            'entropy_coeff': 0.05,
            'sweep_type': 'random',
            'n_random_hparams': 20,
            'steps_log_freq': 16,
            'update_log_freq': 20,
            'total_steps': int(1e7),
            'seed': 2024,
            'n_seeds': 5,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
