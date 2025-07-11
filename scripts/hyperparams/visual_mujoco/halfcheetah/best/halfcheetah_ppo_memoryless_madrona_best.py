from pathlib import Path

exp_name = Path(__file__).stem

lrs = [0.000025]
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
            'env': 'halfcheetah_pixels',
            'double_critic': False,
            'memoryless': True,
            'action_concat': False,
            'lr': lrs,
            'lambda0': lambda0s,
            'lambda1': ' '.join(map(str, lambda1s)),
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'hidden_size': 512,
            'entropy_coeff': 0.01,
            'steps_log_freq': 8,
            'update_log_freq': 10,
            'total_steps': int(5e6),
            'seed': [2050 + i for i in range(30)],
            'n_seeds': 1,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}
