from pathlib import Path

exp_name = Path(__file__).stem

lrs = [2.5e-3, 2.5e-4, 2.5e-5, 2.5e-6]
lambda0s = [0.1, 0.3, 0.5, 0.7, 0.9, 0.95]
lambda1s = [0.95]
alphas = [1]
ld_weights = [0]
rnd_lrs = [2.5e-4]

hparams = {
    'file_name':
        f'runs_{exp_name}.txt',
    'entry': '-m pobax.algos.ppo_memory_rnd',
    'args': [
        {
            'env': 'rocksample_15_15',
            'double_critic': False,
            'action_concat': True,
            'lr': lrs,
            'lambda0': lambda0s,
            'lambda1': ' '.join(map(str, lambda1s)),
            'alpha': ' '.join(map(str, alphas)),
            'ld_weight': ' '.join(map(str, ld_weights)),
            'rnd_lr': ' '.join(map(str, rnd_lrs)), 
            'rnd_loss_coeff': 1.0,
            'rnd_gae_coeff': 0.1,
            'rnd_reward_coeff': 1.0,
            'rnd_hidden_size': 512,
            'hidden_size': 512,
            'num_envs': 16,
            'entropy_coeff': 0.2,
            'steps_log_freq': 8,
            'update_log_freq': 10,
            'total_steps': int(1e7),
            'seed': 2024,
            'n_seeds': 5,
            'platform': 'gpu',
            'study_name': exp_name
        }
    ]
}