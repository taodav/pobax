from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5, 2.5e-6]
lambda0s = [0.1, 0.5, 0.7, 0.9, 0.95]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.transformer_xl',
    'args': [{
        'env': 'tmaze_5',
        'double_critic': False,
        'lr': lrs,
        'lambda0': lambda0s,
        'lambda1': ' '.join(map(str, lambda1s)),
        'alpha': ' '.join(map(str, alphas)),
        'ld_weight': ld_weights,
        'entropy_coeff': 0.1,
        'hidden_size': 32,
        'embed_size': 16,
        'total_steps': int(1e6),
        'steps_log_freq': 8,
        'update_log_freq': 10,
        'seed': 2024,
        'debug': True,
        'n_seeds': 10,
        'platform': 'gpu',
        'study_name': exp_name
    }]
}