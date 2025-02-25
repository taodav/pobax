from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5, 2.5e-6]
lambda0s = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [
        {
            'env': 'rocksample_11_11',
            'action_concat': True,
            'perfect_memory': True,
            'memoryless': False,
            'lr': ' '.join(map(str, lrs)),
            'lambda0': ' '.join(map(str, lambda0s)),
            'hidden_size': [32, 64, 128],
            'num_envs': 8,
            'entropy_coeff': 0.2,
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
