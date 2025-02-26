from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-4]
lambda0s = [0.7]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [{
        'env': 'Hopper-V-v0',
        'action_concat': True,
        'double_critic': False,
        'lr': ' '.join(map(str, lrs)),
        'lambda0': lambda0s,
        'lambda1': ' '.join(map(str, lambda1s)),
        'alpha': ' '.join(map(str, alphas)),
        'hidden_size': 256,
        'steps_log_freq': 16,
        'update_log_freq': 20,
        'total_steps': int(5e7),
        'seed': 2025,
        'n_seeds': 30,
        'platform': 'gpu',
        'study_name': exp_name
    }]
}
