from pathlib import Path

import flax.linen as nn
import gymnax
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import orbax.checkpoint
import scipy as scp


from pobax.algos.dqn import QNetwork, TimeStep


class FeatureQNetwork(QNetwork):
    @nn.compact
    def __call__(self, x: jnp.ndarray):
        x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        features = nn.relu(x)
        q_vals = nn.Dense(self.action_dim)(features)
        return q_vals, features


def load_data_from_dict(dict_bstate: dict) -> TimeStep:
    dict_exp = jax.tree.map(lambda x: jnp.array(x)[:, 0], dict_bstate['experience'])
    tstep = TimeStep(**dict_exp)
    return tstep


def make_calculate(config: dict):
    basic_env, env_params = gymnax.make(config["ENV_NAME"])
    network = FeatureQNetwork(action_dim=basic_env.action_space(env_params).n, hidden_size=config["HIDDEN_SIZE"])
    gamma = config['GAMMA']

    def get_errors(params, dataset):
        # get the features of our dataset
        _, all_features = network.apply(params, dataset.obs)

        # parse our dataset
        features, next_features = all_features[:-1], all_features[1:]
        dones = dataset.done[:-1]
        not_dones = (1 - dones).astype(bool)
        rewards = dataset.reward[:-1]

        # initialize our AtA terms (incl. the bias term)
        bias = jnp.ones((features.shape[0], 1))

        features_with_bias = jnp.concatenate((features, bias), axis=-1)
        next_features_with_bias = jnp.concatenate((next_features, bias), axis=-1)

        features_with_bias_excl_dones = features_with_bias * not_dones[..., None]
        # next_features_with_bias_excl_dones = next_features_with_bias * not_dones[..., None]

        # For reward prediction
        A_t_A = jnp.matmul(features_with_bias.T, features_with_bias)
        A_t_A += jnp.eye(A_t_A.shape[0]) * epsilon
        A_t_R = jnp.matmul(features_with_bias.T, rewards)

        # For next feature prediction
        A_t_A_excl_done = jnp.matmul(features_with_bias_excl_dones.T, features_with_bias_excl_dones)
        A_t_A_excl_done += jnp.eye(A_t_A_excl_done.shape[0]) * epsilon
        A_t_nA_excl_done = jnp.matmul(features_with_bias_excl_dones.T, next_features * not_dones[..., None])

        # For LSTD
        exp_disc_next_feature_diffs = (features_with_bias - gamma * (1 - dones[..., None]) * next_features_with_bias)
        TD_A_t_A = jnp.matmul(exp_disc_next_feature_diffs.T, features_with_bias)
        TD_A_t_A += jnp.eye(TD_A_t_A.shape[0]) * epsilon

        # Now we solve for our fixed points
        R_phi = jnp.linalg.solve(A_t_A, A_t_R)

        P_phi = jnp.linalg.solve(A_t_A_excl_done, A_t_nA_excl_done)

        w_phi = jnp.linalg.solve(TD_A_t_A, A_t_R)

        # and get all of our predictions & errors
        R_preds = jnp.matmul(features_with_bias, R_phi)
        R_err = R_preds - rewards

        P_preds = jnp.matmul(features_with_bias_excl_dones, P_phi)
        P_err = P_preds - next_features * not_dones[..., None]
        # P_err_with_bias = jnp.concatenate((P_err, jnp.ones((P_err.shape[0], 1))), axis=-1)

        val_preds = jnp.matmul(features_with_bias, w_phi)
        next_val_preds = jnp.matmul(next_features_with_bias, w_phi)
        val_err = rewards + gamma * next_val_preds - val_preds

        results = {
            'R_err': R_err, 'P_err': P_err, 'val_err': val_err,
            'R_phi': R_phi, 'P_phi': P_phi, 'w_phi': w_phi
        }

        return results

    return get_errors


if __name__ == "__main__":
    dataset_path = Path('/Users/ruoyutao/Documents/pobax/results/Acrobot-v1_2024_37b0becb6279b2b7e878de87971959e9_buffer_2024')
    seed = 2024
    epsilon = 1e-7

    key = jax.random.PRNGKey(seed)

    # load our params
    orbax_checkpointer = orbax.checkpoint.PyTreeCheckpointer()
    restored = orbax_checkpointer.restore(dataset_path)
    config = restored['config']

    datasets = load_data_from_dict(restored['buffer_state'])  # TimeStep object
    paramses = restored['params']

    get_errors_fn = make_calculate(config)

    # # TODO: Debugging, remove later.
    # # We first try to do "the thing" with the last checkpoint.
    # params = jax.tree.map(lambda x: x[-1], paramses)
    # dataset = jax.tree.map(lambda x: x[-1], datasets)
    #
    # error_dict = get_errors_fn(params, dataset)

    # vmap the get errors fn
    vmap_get_errors = jax.jit(jax.vmap(get_errors_fn, in_axes=0))
    error_dict = vmap_get_errors(paramses, datasets)

    fig, ax = plt.subplots(1, 1)

    diff_per_feat_err = (error_dict['val_err'] - error_dict['R_err'])
    mean_diff_per_feat_err = diff_per_feat_err.mean(axis=-1)
    std_err_diff_per_feat_err = scp.stats.sem(diff_per_feat_err, axis=-1)

    x = jnp.arange(mean_diff_per_feat_err.shape[0])
    ax.plot(x, mean_diff_per_feat_err, color='blue', label='val_err - R_err')
    ax.fill_between(x, mean_diff_per_feat_err - std_err_diff_per_feat_err, mean_diff_per_feat_err + std_err_diff_per_feat_err,
                      color='blue', alpha=0.35)

    mean_reward_err = error_dict['R_err'].mean(axis=-1)
    std_err_reward_err = scp.stats.sem(error_dict['R_err'], axis=-1)
    ax.plot(x, mean_reward_err, color='orange', label='R_err')
    ax.fill_between(x, mean_reward_err - std_err_reward_err, mean_reward_err + std_err_reward_err,
                    color='orange', alpha=0.35)

    plt.legend(loc='lower right')
    plt.show()

    print()
