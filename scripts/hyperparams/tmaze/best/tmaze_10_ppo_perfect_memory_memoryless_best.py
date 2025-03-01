from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-4]
lambda0s = [0.5]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [
        {
            'env': 'tmaze_10',
            'action_concat': False,
            'perfect_memory': True,
            'memoryless': True,
            'lr': ' '.join(map(str, lrs)),
            'lambda0': ' '.join(map(str, lambda0s)),
            'hidden_size': 32,
            'num_envs': 4,
            'entropy_coeff': 0.01,
            'steps_log_freq': 4,
            'update_log_freq': 5,
            'total_steps': int(1e6),
            'seed': 2025,
            'n_seeds': 30,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
