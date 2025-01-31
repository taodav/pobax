from pathlib import Path
import importlib

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5, 2.5e-6]
lambda0s = [0.95]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo',
    'args': [{
        'env': [
            'Pendulum-P-v0', 'Hopper-P-v0', 'Walker-P-v0', 'Ant-P-v0', 'HalfCheetah-P-v0',
            'Pendulum-V-v0', 'Hopper-V-v0', 'Walker-V-v0', 'Ant-V-v0', 'HalfCheetah-V-v0',
        ],
        'double_critic': False,
        'memoryless': True,
        'lr': ' '.join(map(str, lrs)),
        'lambda0': ' '.join(map(str, lambda0s)),
        'lambda1': ' '.join(map(str, lambda1s)),
        'alpha': ' '.join(map(str, alphas)),
        'ld_weight': ' '.join(map(str, ld_weights)),
        'hidden_size': 256,
        'steps_log_freq': 16,
        'update_log_freq': 20,
        'total_steps': int(5e7),
        'seed': 2025,
        'n_seeds': 5,
        'platform': 'gpu',
        'study_name': exp_name
    }]
}
