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
            'env': 'battleship_10',
            'action_concat': False,
            'perfect_memory': True,
            'memoryless': False,
            'lr': lrs,
            'lambda0': lambda0s,
            'hidden_size': 512,
            'num_envs': 32,
            'entropy_coeff': 0.05,
            'steps_log_freq': 8,
            'update_log_freq': 10,
            'total_steps': int(1e7),
            'seed': 2024,
            'n_seeds': 10,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
