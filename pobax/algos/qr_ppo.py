import jax.numpy as jnp

from pobax.algos.ppo import PPO


class QRPPO(PPO):
    def __init__(self, network,
                 double_critic: bool = False,
                 ld_weight: float = 0.,
                 alpha: float = 0.,
                 vf_coeff: float = 0.,
                 ld_exploration_bonus_scale: float = 0.,
                 entropy_coeff: float = 0.01,
                 clip_eps: float = 0.2,
                 n_atoms: int = 51,
                 kappa: float = 1.):
        super().__init__(network, double_critic, ld_weight, alpha, vf_coeff,
                         ld_exploration_bonus_scale, entropy_coeff, clip_eps)
        self.n_atoms = n_atoms
        self.atoms = jnp.arange(self.n_atoms)
        self.kappa = kappa

    def loss(self, params, init_hstate, traj_batch, gae, targets):
        # RERUN NETWORK
        _, pi, quantiles = self.network.apply(
            params, init_hstate[0], (traj_batch.obs, traj_batch.done)
        )
        log_prob = pi.log_prob(traj_batch.action)

        # CALCULATE QUANTILE VALUE LOSS
        quantiles_clipped = traj_batch.value + (
                quantiles - traj_batch.value
        ).clip(-self.clip_eps, self.clip_eps)

        td_err = targets[..., None, :] - quantiles[..., None]
        clipped_td_err = targets[..., None, :] - quantiles_clipped[..., None]  # b x n_atoms x n_atoms

        def loss_over_err(err: jnp.ndarray):
            huber_loss = ((jnp.abs(err) <= self.kappa).astype(float) * 0.5 * (err ** 2)
                          + (jnp.abs(err) > self.kappa).astype(float) * self.kappa
                          * (jnp.abs(err) - 0.5 * self.kappa)
                          )

            # get quantile mid points
            tau_hat = (self.atoms + 0.5) / self.atoms.shape[0]
            all_idxes = list(range(len(err.shape)))
            dims_to_expand = [i for i in all_idxes if i != all_idxes[-2]]
            tau_hat_expanded = jnp.expand_dims(tau_hat, dims_to_expand)
            tau_bellman_diff = jnp.abs(
                tau_hat_expanded - (err < 0).astype(float)
            )
            quantile_huber_loss = tau_bellman_diff * huber_loss
            loss = jnp.sum(jnp.mean(quantile_huber_loss, axis=-1), axis=-1)
            return loss

        value_loss = loss_over_err(td_err)
        clipped_value_loss = loss_over_err(clipped_td_err)
        value_loss = (
            jnp.maximum(value_loss, clipped_value_loss).mean()
        )

        # Quantile lambda discrepancy loss.
        # the mean l2 distance between quantiles is the Wasserstein-2 distance, as per
        # https://stats.stackexchange.com/questions/465229/measuring-the-distance-between-two-probability-measures-using-quantile-functions
        if self.double_critic:
            wasserstein_2_dist = (jnp.square(quantiles[..., 0, :] - quantiles[..., 1, :])).mean()
            value_loss = self.ld_weight * wasserstein_2_dist + \
                         (1 - self.ld_weight) * value_loss

        # CALCULATE ACTOR LOSS
        ratio = jnp.exp(log_prob - traj_batch.log_prob)

        # which advantage do we use to update our policy?
        if self.double_critic:
            gae = (self.alpha * gae[..., 0] +
                   (1 - self.alpha) * gae[..., 1])
        gae = (gae - gae.mean()) / (gae.std() + 1e-8)
        loss_actor1 = ratio * gae
        loss_actor2 = (
                jnp.clip(
                    ratio,
                    1.0 - self.clip_eps,
                    1.0 + self.clip_eps,
                    )
                * gae
        )
        loss_actor = -jnp.minimum(loss_actor1, loss_actor2)
        loss_actor = loss_actor.mean()
        entropy = pi.entropy().mean()

        total_loss = (
                loss_actor
                + self.vf_coeff * value_loss
                - self.entropy_coeff * entropy
        )
        return total_loss, (value_loss, loss_actor, entropy)


