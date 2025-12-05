import math
import jax
import equinox as eqx
from equinox import nn
import jax.numpy as jnp
import optax


class Block(eqx.Module):
    net: eqx.Module
    """A layer of MLP with layer norm and leaky relu"""

    def __init__(self, input_size, output_size, key):
        self.net = nn.Sequential(
            [
                nn.Linear(input_size, output_size, key=key),
                nn.LayerNorm(output_size),
                nn.Lambda(jax.nn.leaky_relu),
            ]
        )

    def __call__(self, x, key=None):
        return self.net(x, key=key)


def default_init(key, linear, scale=1.0, zero_bias=False, fixed_bias=None):
    # eqx's default is 1/sqrt(in_features)
    lim = math.sqrt(scale / linear.in_features)
    linear = eqx.tree_at(
        lambda l: l.weight,
        linear,
        jax.random.uniform(key, linear.weight.shape, minval=-lim, maxval=lim),
    )
    if zero_bias:
        linear = eqx.tree_at(lambda l: l.bias, linear, jnp.zeros_like(linear.bias))
    elif fixed_bias is not None:
        linear = eqx.tree_at(
            lambda l: l.bias, linear, jnp.full_like(linear.bias, fixed_bias)
        )
    return linear


def final_linear(key, input_size, output_size, scale=0.01):
    linear = nn.Linear(input_size, output_size, key=key)
    linear = default_init(key, linear, scale=scale)
    linear = eqx.tree_at(lambda l: l.bias, linear, linear.bias * 0.0)
    return linear


def huber(x):
    return jax.lax.select(jnp.abs(x) < 1.0, 0.5 * x**2, jnp.abs(x) - 0.5)


def soft_update(network, target, tau):
    def polyak(param, target_param):
        return target_param * (1 - tau) + param * tau

    params, _ = eqx.partition(network, eqx.is_inexact_array)
    target_params, static = eqx.partition(target, eqx.is_inexact_array)
    updated_params = jax.tree.map(polyak, params, target_params)
    target = eqx.combine(static, updated_params)
    target = eqx.nn.inference_mode(target, True)
    return target


@eqx.filter_jit
def feature_metrics(features: jnp.ndarray):
    # features: (B, d_hidden) where B >= d_hidden
    # adapted from https://github.com/RLE-Foundation/Plasticine
    singular_values = jnp.linalg.svdvals(features)
    cumsum_singular_values = jnp.cumsum(singular_values) / jnp.sum(singular_values)
    stable_rank = jnp.sum(cumsum_singular_values < 0.99) + 1
    feature_norm = jnp.mean(jnp.linalg.norm(features, axis=-1))

    return {
        "stable_rank": stable_rank,
        "feature_norm": feature_norm,
    }


def get_optimizer(lr: float, lr_ratio: float, config: dict):
    lr_schedule = optax.schedules.cosine_decay_schedule(
        init_value=lr,
        decay_steps=config["grad_steps"],
        alpha=lr_ratio,
    )
    return optax.chain(
        optax.zero_nans(),
        optax.clip_by_global_norm(config["gradient_scale"]),
        optax.inject_hyperparams(optax.adamw)(
            learning_rate=lr_schedule,
            weight_decay=config["weight_decay"],
            eps=config["adam_eps"],
        ),
    )
