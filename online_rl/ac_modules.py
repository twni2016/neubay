from typing import Any, Dict, List
import jax
import equinox as eqx
from equinox import nn
from jax import random
import jax.numpy as jnp
from online_rl.common import Block, final_linear
import distrax
from typing import Dict, Any, Optional
from jaxtyping import PyTree


class CriticHead(eqx.Module):
    layers: List[eqx.Module]
    out: nn.Linear

    def __init__(self, input_size, hidden_size, n_layers, key):
        keys = random.split(key, n_layers + 1)
        self.layers = []
        assert n_layers >= 1
        for i in range(n_layers):
            in_size = input_size if i == 0 else hidden_size
            self.layers.append(eqx.filter_vmap(Block(in_size, hidden_size, keys[i])))

        self.out = eqx.filter_vmap(final_linear(keys[-1], hidden_size, 1, scale=0.01))

    def __call__(self, x, key):
        T = x.shape[0]
        net_keys = random.split(key, len(self.layers) * T)
        for i in range(len(self.layers)):
            x = self.layers[i](x, net_keys[i * T : (i + 1) * T])

        qs = self.out(x).squeeze(-1)
        return qs


class RecurrentCritic(eqx.Module):
    """The core model used in experiments. Optimizer depends on its field names."""

    input_size: int
    action_dim: int
    config: Dict[str, Any]
    pre: eqx.Module
    memory: eqx.Module
    critic_heads: eqx.Module

    def __init__(self, obs_shape, act_shape, memory_module, config, key):
        self.memory = memory_module

        self.config = config
        [self.input_size] = obs_shape
        [self.action_dim] = act_shape

        keys = random.split(key, 2)
        self.pre = eqx.filter_vmap(Block(self.input_size, self.memory.d_model, keys[0]))

        @eqx.filter_vmap
        def _make_ensemble(k):
            return CriticHead(
                input_size=self.memory.d_model + self.action_dim,
                hidden_size=config["head_d_hidden"],
                n_layers=config["head_n_layers"],
                key=k,
            )

        assert config["num_critics"] >= config["sampled_critics"]
        critic_keys = random.split(keys[1], config["num_critics"])
        self.critic_heads = _make_ensemble(critic_keys)

    @eqx.filter_jit
    def __call__(self, x, state, start, action, key):
        if x.ndim == 2:
            """
            Training: a long tape of several episodes
            x: (T, d_input) where T is tape length
            state: L[(1, d_hidden)] complex
            start: (T,) bool
            """
            pre_key, mem_key, key = random.split(key, 3)
            T = x.shape[0]
            x = self.pre(x, random.split(pre_key, T))  # x: (T, d_model)
            y, _ = self.memory(x=x, state=state, start=start, key=mem_key)
            # y: encoder's output (T, d_model) real
        else:
            """
            Evaluation: batched rollouts
            x: (B, T, d_input)
            state, new_state: L[(B, 1, d_hidden)] complex
            start: (B, T) bool
            """
            B, T, _ = x.shape
            pre_key, mem_key = random.split(key, 2)
            pre_keys = random.split(pre_key, B * T)
            mem_keys = random.split(mem_key, B)

            x_flat = x.reshape(B * T, -1)
            x_flat = self.pre(x_flat, pre_keys)  # (B*T, d_model)
            x_bt = x_flat.reshape(B, T, -1)  # (B, T, d_model)

            # vmap on the batch-axis of all leaves in the inputs
            y_bt, _ = jax.vmap(self.memory)(x_bt, state, start, mem_keys)
            y = y_bt[:, -1]  # (B, d_model)

        """
        y: (*, d_model)
        action: (*, d_action)
        """
        y_act = jnp.concatenate([y, action], axis=-1)

        @eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None))
        def ensemble(model, x, key):
            return model(x, key=key)

        q = ensemble(self.critic_heads, y_act, key)  # (N, *)
        return q

    def initial_state(self, shape=tuple()):
        if isinstance(shape, int):
            shape = (shape,)
        return self.memory.initial_state(shape)


def _select_tree(tree, idx: jax.Array):
    """Return a copy of `tree` with leaves[axis0] restricted to idx."""

    def _maybe_index(x):
        if isinstance(x, jnp.ndarray) and x.ndim > 0:
            return x[idx]  # keep selected heads
        else:
            return x  # non‑array or non‑stacked leaf

    return jax.tree.map(_maybe_index, tree)


def select_critics(model: RecurrentCritic, idx: jax.Array) -> RecurrentCritic:
    """Return a new model whose `critic_heads` is restricted to `idx`."""
    new_heads = _select_tree(model.critic_heads, idx)
    return eqx.tree_at(lambda m: m.critic_heads, model, new_heads)


"""
SAC policy
Below is adapted from https://github.com/Howuhh/sac-n-jax/blob/main/sac_n_jax_eqx.py
Unfortunately, distrax is not compatible by default with equinox, so some hacks are needed
see: https://github.com/patrick-kidger/equinox/issues/269#issuecomment-1446586093
"""


class TanhNormal(distrax.Transformed):
    def __init__(self, loc, scale):
        normal_dist = distrax.Normal(loc, scale)
        tanh_bijector = distrax.Tanh()
        super().__init__(distribution=normal_dist, bijector=tanh_bijector)

    def mode(self):
        return self.bijector.forward(self.distribution.mode())


class FixedDistrax(eqx.Module):
    cls: type
    args: PyTree[Any]
    kwargs: PyTree[Any]

    def __init__(self, cls, *args, **kwargs):
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def sample_and_log_prob(self, *, seed):
        return self.cls(*self.args, **self.kwargs).sample_and_log_prob(seed=seed)

    def sample(self, *, seed):
        return self.cls(*self.args, **self.kwargs).sample(seed=seed)

    def log_prob(self, x):
        return self.cls(*self.args, **self.kwargs).log_prob(x)

    def mode(self):
        return self.cls(*self.args, **self.kwargs).mode()


class ActorHead(eqx.Module):
    action_dim: int
    layers: List[eqx.Module]
    out: nn.Linear

    def __init__(self, act_shape, feature_dim, hidden_size, n_layers, key):
        [self.action_dim] = act_shape
        keys = random.split(key, n_layers + 1)
        self.layers = []
        assert n_layers >= 1
        for i in range(n_layers):
            in_size = feature_dim if i == 0 else hidden_size
            self.layers.append(eqx.filter_vmap(Block(in_size, hidden_size, keys[i])))

        self.out = eqx.filter_vmap(
            nn.Linear(hidden_size, self.action_dim * 2, key=keys[-1])
        )

    def __call__(self, x, key):
        T = x.shape[0]
        net_keys = random.split(key, len(self.layers) * T)
        for i in range(len(self.layers)):
            x = self.layers[i](x, net_keys[i * T : (i + 1) * T])

        mu, log_sigma = jnp.split(self.out(x), 2, axis=-1)  # (B, d_action)
        # clipping params from EDAC paper, not as in SAC paper (-20, 2)
        sigma = jnp.exp(jnp.clip(log_sigma, -5, 2))

        dist = FixedDistrax(TanhNormal, mu, sigma)
        return dist


class RecurrentActor(eqx.Module):
    """The core model used in experiments. Optimizer depends on its field names."""

    input_size: int
    config: Dict[str, Any]
    pre: eqx.Module
    memory: eqx.Module
    actor_head: eqx.Module

    def __init__(self, obs_shape, act_shape, memory_module, config, key):
        self.memory = memory_module

        self.config = config
        [self.input_size] = obs_shape

        keys = random.split(key, 2)
        self.pre = eqx.filter_vmap(Block(self.input_size, self.memory.d_model, keys[0]))

        self.actor_head = ActorHead(
            act_shape,
            feature_dim=self.memory.d_model,
            hidden_size=config["head_d_hidden"],
            n_layers=config["head_n_layers"],
            key=keys[1],
        )

    @eqx.filter_jit
    def __call__(self, x, state, start, key, deterministic: bool = False):
        if x.ndim == 2:
            """
            Training: a long tape of several episodes
            x: (T, d_input) where T is tape length
            state: L[(1, d_hidden)] complex
            start: (T,) bool
            """
            pre_key, mem_key, pi_key = random.split(key, 3)
            T = x.shape[0]
            x = self.pre(x, random.split(pre_key, T))  # x: (T, d_model)
            y, _ = self.memory(x=x, state=state, start=start, key=mem_key)
            # y: encoder's output (T, d_model) real

            dist = self.actor_head(y, key=pi_key)
            return dist
        else:
            """
            Evaluation: batched rollouts
            x: (B, T, d_input)
            state, new_state: L[(B, 1, d_hidden)] complex
            start: (B, T) bool
            """
            B, T, _ = x.shape
            pre_key, mem_key, pi_key, act_key = random.split(key, 4)
            pre_keys = random.split(pre_key, B * T)
            mem_keys = random.split(mem_key, B)

            x_flat = x.reshape(B * T, -1)
            x_flat = self.pre(x_flat, pre_keys)  # (B*T, d_model)
            x_bt = x_flat.reshape(B, T, -1)  # (B, T, d_model)

            # vmap on the batch-axis of all leaves in the inputs
            y_bt, new_state = jax.vmap(self.memory)(x_bt, state, start, mem_keys)
            y = y_bt[:, -1]  # (B, d_model)

            dist = self.actor_head(y, key=pi_key)  # over (B, d_action)

            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample(seed=act_key)
            return action, new_state, y

    def initial_state(self, shape=tuple()):
        if isinstance(shape, int):
            shape = (shape,)
        return self.memory.initial_state(shape)


class Alpha(eqx.Module):
    value: jnp.float32  # param
    target_entropy: Optional[float]  # constant, if None, use fixed alpha

    def __init__(self, init_value, target_entropy):
        self.target_entropy = target_entropy
        self.value = jnp.log(jnp.float32(init_value))

    def __call__(self):
        return jnp.exp(self.value)
