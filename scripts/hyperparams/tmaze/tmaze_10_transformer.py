from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5, 2.5e-6]
lambda0s = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.transformer_xl',
    'args': [
        {
            'env': 'tmaze_10',
            'double_critic': False,
            'action_concat': True,
            'lr': lrs,
            'lambda0': lambda0s,
            'lambda1': ' '.join(map(str, lambda1s)),
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'hidden_size': 32,
            'embed_size': 12,
            'num_envs': 4,
            'entropy_coeff': 0.01,
            'steps_log_freq': 4,
            'update_log_freq': 5,
            'total_steps': int(1e6),
            'seed': 2024,
            'n_seeds': 10,
            'debug': True,
            'show_discounted': True,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}