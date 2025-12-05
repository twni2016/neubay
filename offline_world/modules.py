import jax
import jax.numpy as jnp
import equinox as eqx
from equinox import nn
import numpy as np


class Scaler:
    def __init__(self, observations, actions, rewards):
        obs_mean = np.atleast_1d(np.mean(observations, axis=0))
        obs_std = np.atleast_1d(np.std(observations, axis=0))
        act_mean = np.atleast_1d(np.mean(actions, axis=0))
        act_std = np.atleast_1d(np.std(actions, axis=0))
        rew_mean = np.atleast_1d(np.mean(rewards, axis=0))
        rew_std = np.atleast_1d(np.std(rewards, axis=0))

        self.input_mean = np.concatenate([obs_mean, act_mean])
        self.input_std = np.concatenate([obs_std, act_std])
        self.input_std[self.input_std < 1e-12] = 1.0

        self.output_mean = np.concatenate([obs_mean, rew_mean])
        self.output_std = np.concatenate([obs_std, rew_std])
        self.output_std[self.output_std < 1e-12] = 1.0

    def transform_inputs(self, x):
        return (x - self.input_mean) / self.input_std

    def transform_outputs(self, y):
        return (y - self.output_mean) / self.output_std

    def inverse_transform_outputs(self, y):
        return y * self.output_std + self.output_mean


def update_selected_members(dst, src, mask: jax.Array):
    """Return a new PyTree equal to `dst` except that, for every array leaf
    whose leading axis is the ensemble axis, the entries where `mask` is True
    are replaced by those from `src`. mask: (N) boolean array"""

    def _maybe_replace(d, s):
        if isinstance(d, jnp.ndarray) and d.ndim > 0 and d.shape[0] == mask.shape[0]:
            return d.at[mask].set(s[mask])
        else:  # scalars, callables, etc.
            return d

    return jax.tree.map(_maybe_replace, dst, src)


def subselect_members(tree, idx: jax.Array):
    """Return a copy of `tree` with leaves[axis0] restricted to idx."""
    idx = jnp.asarray(idx)

    def _maybe_index(x):
        if isinstance(x, jnp.ndarray) and x.ndim > 0 and x.shape[0] >= idx.max() + 1:
            return x[idx]  # keep selected heads
        else:
            return x  # non‑array or non‑stacked leaf

    return jax.tree.map(_maybe_index, tree)


def soft_clamp(
    x: jax.Array,
    _min: float,
    _max: float,
) -> jax.Array:
    """
    Clamp tensor values while maintaining the gradient.
    Reference: PETS https://arxiv.org/abs/1805.12114
    https://github.com/yihaosun1124/OfflineRL-Kit/blob/main/offlinerlkit/modules/dynamics_module.py
    """
    x = _max - jax.nn.softplus(_max - x)  # < min(_max, x)
    x = _min + jax.nn.softplus(x - _min)  # > max(_min, x)
    return x


class ModelBlock(eqx.Module):
    net: eqx.Module
    """A layer of MLP with layer norm and leaky relu"""

    def __init__(self, input_size, output_size, has_ln: bool, key):
        if has_ln:
            self.net = nn.Sequential(
                [
                    nn.Linear(input_size, output_size, key=key),
                    nn.LayerNorm(output_size),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )
        else:
            self.net = nn.Sequential(
                [
                    nn.Linear(input_size, output_size, key=key),
                    nn.Lambda(jax.nn.leaky_relu),
                ]
            )

    def __call__(self, x):
        x = self.net(x)
        return x


class BanditModel(eqx.Module):
    block1: eqx.Module
    block2: eqx.Module
    mu_layer: eqx.Module
    logsigma_layer: eqx.Module
    min_logsigma: float = -5
    max_logsigma: float = 0.25

    def __init__(self, input_size, hidden_size, has_ln, key):
        keys = jax.random.split(key, 4)
        self.block1 = eqx.filter_vmap(
            ModelBlock(input_size, hidden_size, has_ln, keys[0])
        )
        self.block2 = eqx.filter_vmap(
            ModelBlock(hidden_size, hidden_size, has_ln, keys[1])
        )
        self.mu_layer = eqx.filter_vmap(eqx.nn.Linear(hidden_size, 1, key=keys[2]))
        self.logsigma_layer = eqx.filter_vmap(
            eqx.nn.Linear(hidden_size, 1, key=keys[3])
        )
        # with the default initialization (hidden size=16),
        # mu is initialized round [-1, 1], std is around [0.1, 1]

    def __call__(self, x):
        x = self.block1(x)
        x = self.block2(x)
        mu = self.mu_layer(x)
        logsigma = soft_clamp(
            self.logsigma_layer(x), self.min_logsigma, self.max_logsigma
        )
        return mu, jnp.exp(logsigma)


class EnsembleBanditModel(eqx.Module):
    members: eqx.Module  # every weight now has shape (N, …)

    def __init__(
        self,
        ensemble_size: int,
        input_size: int,
        hidden_size: int,
        has_ln: bool,
        key: jax.Array,
    ):
        member_keys = jax.random.split(key, ensemble_size)

        @eqx.filter_vmap
        def _make_ensemble(k):
            return BanditModel(input_size, hidden_size, has_ln, k)

        self.members = _make_ensemble(member_keys)

    @property
    def ensemble_size(self):
        # the first linear layer
        first_leaf = jax.tree.leaves(self.members)[0]
        return first_leaf.shape[0]

    def forward_same_data(
        self,
        x: jnp.ndarray,
    ):
        """
        x : (B, input_size)
        mu     : (N, B, 1)
        sigma  : (N, B, 1)
        unc    : dict of (B) arrays
            epi_mean: the epistemic term of total variance, but does not count std parameters
                      strictly speaking, it is underestimated https://stats.stackexchange.com/a/11818
            ale_max: exactly MOPO (Yu et al., 2020), MAPLE (Chen et al., 2021)
            total_var: counts both epistemic and aleatoric uncertainties
                     exactly the ensemble stddev used in (Lu et al., 2022)
        """

        # vary parameters along axis 0, share the data
        @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
        def _apply(member, data):
            return member(data)

        mu, sigma = _apply(self.members, x)

        unc = {
            "epi_mean": mu.std(0).squeeze(-1),
            "ale_max": sigma.max(0).squeeze(-1),
            "total_var": jnp.sqrt(mu.var(0) + (sigma**2).mean(0)).squeeze(-1),
        }

        return (mu, sigma), unc

    def forward_diff_data(self, x: jnp.ndarray):
        """
        x : (N, B, input_size)
        mu     : (N, B, 1)
        sigma  : (N, B, 1)
        """

        @eqx.filter_vmap
        def _apply(member, data):
            return member(data)

        mu, sigma = _apply(self.members, x)
        return mu, sigma


class ContModel(eqx.Module):
    obs_dim: int
    block1: eqx.Module
    block2: eqx.Module
    block3: eqx.Module
    block4: eqx.Module
    mu_layer: eqx.Module
    logsigma_layer: eqx.Module
    min_logsigma: float = -5
    max_logsigma: float = 0.25

    def __init__(self, obs_dim, act_dim, hidden_size, has_ln, key):
        keys = jax.random.split(key, 6)
        self.obs_dim = obs_dim
        self.block1 = eqx.filter_vmap(
            ModelBlock(obs_dim + act_dim, hidden_size, has_ln, keys[0])
        )
        self.block2 = eqx.filter_vmap(
            ModelBlock(hidden_size, hidden_size, has_ln, keys[1])
        )
        self.block3 = eqx.filter_vmap(
            ModelBlock(hidden_size, hidden_size, has_ln, keys[2])
        )
        self.block4 = eqx.filter_vmap(
            ModelBlock(hidden_size, hidden_size, has_ln, keys[3])
        )
        self.mu_layer = eqx.filter_vmap(
            eqx.nn.Linear(hidden_size, obs_dim + 1, key=keys[4])
        )
        self.logsigma_layer = eqx.filter_vmap(
            eqx.nn.Linear(hidden_size, obs_dim + 1, key=keys[5])
        )

    def __call__(self, inputs):
        x = self.block1(inputs)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)

        delta_mu = self.mu_layer(x)
        ## predict the delta observation instead
        mu = jnp.concatenate(
            [
                delta_mu[..., :-1] + inputs[..., : self.obs_dim],  # next-obs mean
                delta_mu[..., -1:],
            ],  # reward mean
            axis=-1,
        )

        logsigma = soft_clamp(
            self.logsigma_layer(x), self.min_logsigma, self.max_logsigma
        )
        return mu, jnp.exp(logsigma)


class EnsembleContModel(eqx.Module):
    members: eqx.Module  # every weight now has shape (N, …)

    def __init__(
        self,
        ensemble_size: int,
        obs_dim: int,
        act_dim: int,
        hidden_size: int,
        has_ln: bool,
        key: jax.Array,
    ):
        member_keys = jax.random.split(key, ensemble_size)

        @eqx.filter_vmap
        def _make_ensemble(k):
            return ContModel(obs_dim, act_dim, hidden_size, has_ln, k)

        self.members = _make_ensemble(member_keys)

    @property
    def ensemble_size(self):
        # the first linear layer
        first_leaf = jax.tree.leaves(self.members.block1)[0]
        return first_leaf.shape[0]

    def forward_same_data(
        self,
        x: jnp.ndarray,
    ):
        """
        x : (B, input_size)
        mu     : (N, B, output_size)
        sigma  : (N, B, output_size)
        unc    : dict of (B) arrays
            epi_mean: the epistemic term of total variance, but does not count std parameters
                      strictly speaking, it is underestimated https://stats.stackexchange.com/a/11818
            ale_max: exactly MOPO (Yu et al., 2020), MAPLE (Chen et al., 2021)
            total_var: counts both epistemic and aleatoric uncertainties
                     exactly the ensemble stddev used in (Lu et al., 2022)
        """

        # vary parameters along axis 0, share the data
        @eqx.filter_vmap(in_axes=(eqx.if_array(0), None))
        def _apply(member, data):
            return member(data)

        mu, sigma = _apply(self.members, x)

        unc = {
            "epi_mean": jnp.sqrt(mu.var(0).mean(-1)),  # root-mean-square
            "ale_max": jnp.max(jnp.linalg.norm(sigma, axis=-1), axis=0),
            "total_var": jnp.sqrt(jnp.mean(mu.var(0) + (sigma**2).mean(0), axis=-1)),
        }

        return (mu, sigma), unc

    def forward_diff_data(self, x: jnp.ndarray):
        """
        x : (N, B, input_size)
        mu     : (N, B, output_size)
        sigma  : (N, B, output_size)
        """

        @eqx.filter_vmap
        def _apply(member, data):
            return member(data)

        mu, sigma = _apply(self.members, x)
        return mu, sigma
