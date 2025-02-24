from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2e-4]
lambda0s = [0.8]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.transformer_xl',
    'args': [{
        'env': 'tmaze_05',
        'double_critic': False,
        'lr': lrs,
        'lambda0': lambda0s,
        'lambda1': ' '.join(map(str, lambda1s)),
        'alpha': ' '.join(map(str, alphas)),
        'gamma': 0.99,
        'ld_weight': ld_weights,
        'entropy_coeff': 0.1,
        'hidden_size': 256,
        'embed_size': 256,
        'window_mem': 128,
        'num_envs': 512,
        'window_grad': 64,
        'total_steps': int(2e6),
        'steps_log_freq': 1,
        'update_log_freq': 1,
        'seed': 2024,
        'debug': True,
        'show_discounted': True,
        'n_seeds': 1,
        'platform': 'gpu',
        'study_name': exp_name
    }]
}