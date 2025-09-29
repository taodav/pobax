from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-4]
lambda0s = [0.7]
lambda1s = [0.95]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.transformer_xl',
    'args': [
        {
            'env': 'rocksample_11_11',
            'double_critic': False,
            'action_concat': True,
            'lr': lrs,
            'lambda0': lambda0s,
            'lambda1': ' '.join(map(str, lambda1s)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'hidden_size': 256,
            'embed_size': 96,
            'num_envs': 16,
            'entropy_coeff': 0.1,
            'steps_log_freq': 4,
            'update_log_freq': 5,
            'total_steps': int(5e6),
            'seed': 2024,
            'n_seeds': 30,
            'debug': True,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
