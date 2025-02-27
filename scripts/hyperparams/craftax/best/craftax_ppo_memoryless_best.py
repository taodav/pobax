from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-05]
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
            'env': 'craftax_pixels',
            'double_critic': False,
            'memoryless': True,
            'action_concat': True,
            'lr': lrs,
            'anneal_lr': True,
            'hidden_size': 512,
            'lambda0': lambda0s,
            'lambda1': ' '.join(map(str, lambda1s)),
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'entropy_coeff': 0.01,
            'steps_log_freq': 20,
            'update_log_freq': 16,
            'num_minibatches': 4,
            'num_steps': 64,
            'num_envs': 256,
            'total_steps': int(5e8),
            'seed': [2050 + i for i in range(10)],
            'n_seeds': 3,
            'platform': 'gpu',
            'debug': True,
            'study_name': exp_name
        }
    ]
}