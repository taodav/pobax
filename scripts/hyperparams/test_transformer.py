from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-4]
lambda0s = [0.95]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [
        {
            'env': 'pocman',
            'double_critic': False,
            'action_concat': False,
            'perfect_memory': True,
            'memoryless': True,
            'lr': lrs,
            'lambda0': lambda0s,
            'lambda1': ' '.join(map(str, lambda1s)),
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'hidden_size': 512,
            'num_envs': 256,
            'entropy_coeff': 0.01,
            'steps_log_freq': 8,
            'update_log_freq': 10,
            'total_steps': int(1e6),
            'seed': 2024,
            'n_seeds': 5,
            'debug': True,
            'show_discounted': True,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}
