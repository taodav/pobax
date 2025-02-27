from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-5]
lambda0s = [0.3]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.transformer_xl',
    'args': [
        {
            'env': 'pocman',
            'double_critic': False,
            'action_concat': True,
            'lr': lrs,
            'lambda0': lambda0s,
            'lambda1': ' '.join(map(str, lambda1s)),
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'num_envs': 16,
            'hidden_size': 512,
            'embed_size': 220,
            'entropy_coeff': 0.05,
            'steps_log_freq': 8,
            'update_log_freq': 10,
            'total_steps': int(1e7),
            'seed': [2126 + i for i in range(10)],
            'n_seeds': 3,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}