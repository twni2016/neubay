from typing import Any, Dict, List
import jax
import equinox as eqx
import optax
from jax import random
import jax.numpy as jnp
from typing import Dict, Any, Optional
from online_rl.common import soft_update, huber
from online_rl.ac_modules import CriticHead, ActorHead, select_critics


class MarkovCritic(eqx.Module):
    """Used in ablation study"""

    obs_dim: int
    action_dim: int
    config: Dict[str, Any]
    critic_heads: eqx.Module

    def __init__(self, obs_shape, act_shape, config, key):

        self.config = config
        [self.obs_dim] = obs_shape
        [self.action_dim] = act_shape

        @eqx.filter_vmap
        def _make_ensemble(k):
            return CriticHead(
                input_size=self.obs_dim + self.action_dim,
                hidden_size=config["d_hidden"],
                n_layers=config["n_layers"],
                key=k,
            )

        assert config["num_critics"] >= config["sampled_critics"]
        critic_keys = random.split(key, config["num_critics"])
        self.critic_heads = _make_ensemble(critic_keys)

    @eqx.filter_jit
    def __call__(self, x, action, key):
        """
        x: (B, d_obs)
        action: (B, d_action)
        """
        x_act = jnp.concatenate([x, action], axis=-1)

        @eqx.filter_vmap(in_axes=(eqx.if_array(0), None, None))
        def ensemble(model, x, key):
            return model(x, key=key)

        q = ensemble(self.critic_heads, x_act, key)  # (N, *)
        return q


class MarkovActor(eqx.Module):
    """Used in ablation study"""

    obs_dim: int
    config: Dict[str, Any]
    actor_head: eqx.Module

    def __init__(self, obs_shape, act_shape, config, key):

        self.config = config
        [self.obs_dim] = obs_shape

        self.actor_head = ActorHead(
            act_shape,
            feature_dim=self.obs_dim,
            hidden_size=config["d_hidden"],
            n_layers=config["n_layers"],
            key=key,
        )

    @eqx.filter_jit
    def __call__(self, x, key, eval_mode: bool, deterministic: bool = False):
        """
        x: (B, d_input) where B is batch size
        """
        pi_key, act_key = random.split(key, 2)
        dist = self.actor_head(x, key=pi_key)

        if eval_mode:
            if deterministic:
                action = dist.mode()
            else:
                action = dist.sample(seed=act_key)
            return action
        else:
            return dist


"""
Markov SAC loss
"""


def sac_q_loss(
    q_network,
    q_target,
    pi_network,
    alpha,
    tape,
    gamma,
    real_weight,
    backup_entropy,
    key,
):
    keys = random.split(key, 5)

    ## first get Q(s, a) of shape (N, B)
    qs = q_network(
        tape["observation"],
        tape["action"],
        key=keys[0],
    )

    ## then get Q_tar(s', a') - log pi(a' | s') where a' ~ pi(s')
    next_action_dist = pi_network(
        tape["next_observation"],
        key=keys[1],
        eval_mode=False,
    )
    next_action, next_action_logp = next_action_dist.sample_and_log_prob(seed=keys[2])

    ### sample M target Qs
    sampled_idx = jax.random.choice(
        keys[3],
        q_network.config["num_critics"],
        shape=(q_network.config["sampled_critics"],),
        replace=False,
    )
    sampled_q_target = select_critics(q_target, sampled_idx)

    next_qs = sampled_q_target(
        tape["next_observation"],
        next_action,
        key=keys[4],
    )
    next_q = jnp.min(next_qs, axis=0)  # (M,B) -> (B)

    if backup_entropy:
        next_q = next_q - alpha() * next_action_logp.sum(-1)
    target = jax.lax.stop_gradient(
        tape["next_reward"] + (1.0 - tape["next_terminated"]) * gamma * next_q
    )

    q_diff = qs - jnp.expand_dims(target, 0)
    loss = huber(q_diff).mean(0)  # (B)

    if "real" in tape:  # mix of real and imaginary data
        real_mask = tape["real"]  # (B)
        imag_mask = 1.0 - tape["real"]  # (B)
        real_sum = jnp.clip(real_mask.sum(), min=1)
        imag_sum = jnp.clip(imag_mask.sum(), min=1)
        real_loss = (loss * real_mask).sum() / real_sum
        imag_loss = (loss * imag_mask).sum() / imag_sum

        loss = real_weight * real_loss + (1.0 - real_weight) * imag_loss
        return loss, {
            "loss_q": loss,
            "loss_q_real": real_loss,
            "loss_q_imag": imag_loss,
            "q_mean": qs[0].mean(),
            "real_q_mean": (qs[0] * real_mask).sum() / real_sum,
            "imag_q_mean": (qs[0] * imag_mask).sum() / imag_sum,
        }

    else:  # pure data
        loss = loss.mean()

        return loss, {
            "loss_q": loss,
            "q_mean": qs[0].mean(),
        }


def sac_pi_loss(pi_network, q_network, alpha, tape, real_weight, key):
    # log pi(a | s) - Q(s, a) where a ~ pi(s)
    keys = random.split(key, 3)
    action_dist = pi_network(tape["observation"], key=keys[0], eval_mode=False)
    action, action_logp = action_dist.sample_and_log_prob(seed=keys[1])

    qs = q_network(
        tape["observation"],
        action,
        key=keys[2],
    )  # (N, B)
    q = qs.mean(0)
    loss = alpha() * action_logp.sum(-1) - q
    entropy = -action_logp.sum(-1)

    if "real" in tape:  # mix of real and imaginary data
        real_mask = tape["real"]  # (B)
        imag_mask = 1.0 - tape["real"]  # (B)
        real_sum = jnp.clip(real_mask.sum(), min=1)
        imag_sum = jnp.clip(imag_mask.sum(), min=1)
        real_loss = (loss * real_mask).sum() / real_sum
        imag_loss = (loss * imag_mask).sum() / imag_sum
        loss = real_weight * real_loss + (1.0 - real_weight) * imag_loss
        return loss, {
            "entropy": jax.lax.stop_gradient(entropy),  # this is used for alpha loss
            "loss_pi": loss,
            "loss_pi_real": real_loss,
            "loss_pi_imag": imag_loss,
        }

    else:  # pure data
        loss = loss.mean()
        return loss, {
            "entropy": jax.lax.stop_gradient(entropy),  # this is used for alpha loss
            "loss_pi": loss,
        }


def sac_alpha_loss(alpha, entropy, tape, real_weight):
    if alpha.target_entropy is None:
        loss = entropy  # a placeholder
    else:
        loss = alpha() * (entropy - alpha.target_entropy)

    if "real" in tape:  # mix of real and imaginary data
        real_mask = tape["real"]  # (B)
        imag_mask = 1.0 - tape["real"]  # (B)
        real_sum = jnp.clip(real_mask.sum(), min=1)
        imag_sum = jnp.clip(imag_mask.sum(), min=1)
        real_loss = (loss * real_mask).sum() / real_sum
        imag_loss = (loss * imag_mask).sum() / imag_sum
        loss = real_weight * real_loss + (1.0 - real_weight) * imag_loss
        return loss, {
            "alpha": jax.lax.stop_gradient(alpha()),
            "real_pi_entropy": (entropy * real_mask).sum() / real_sum,
            "imag_pi_entropy": (entropy * imag_mask).sum() / imag_sum,
        }
    else:
        loss = loss.mean()
        return loss, {
            "alpha": jax.lax.stop_gradient(alpha()),
            "pi_entropy": entropy.mean(),
        }


@eqx.filter_jit
def sac_update(
    q_network,
    pi_network,
    alpha,
    q_target,
    data,
    opt_q,
    opt_pi,
    opt_alpha,
    opt_state_q,
    opt_state_pi,
    opt_state_alpha,
    gamma,
    tau,
    real_weight,
    backup_entropy,
    loss_key,
):
    keys = random.split(loss_key, 2)
    (_, outputs_q), gradient_q = eqx.filter_value_and_grad(sac_q_loss, has_aux=True)(
        q_network,
        q_target,
        eqx.nn.inference_mode(pi_network),
        eqx.nn.inference_mode(alpha),
        data,
        gamma,
        real_weight,
        backup_entropy,
        keys[0],
    )
    updates_q, opt_state_q = opt_q.update(
        gradient_q, opt_state_q, params=eqx.filter(q_network, eqx.is_inexact_array)
    )
    q_network = eqx.apply_updates(q_network, updates_q)
    q_target = soft_update(q_network, q_target, tau=tau)

    (_, outputs_pi), gradient_pi = eqx.filter_value_and_grad(sac_pi_loss, has_aux=True)(
        pi_network,
        eqx.nn.inference_mode(q_network),
        eqx.nn.inference_mode(alpha),
        data,
        real_weight,
        keys[1],
    )
    updates_pi, opt_state_pi = opt_pi.update(
        gradient_pi, opt_state_pi, params=eqx.filter(pi_network, eqx.is_inexact_array)
    )
    pi_network = eqx.apply_updates(pi_network, updates_pi)

    entropy = outputs_pi.pop("entropy")
    if alpha.target_entropy is None:
        _, outputs_alpha = sac_alpha_loss(alpha, entropy, data, real_weight)
    else:
        (_, outputs_alpha), gradient_alpha = eqx.filter_value_and_grad(
            sac_alpha_loss, has_aux=True
        )(alpha, entropy, data, real_weight)
        updates_alpha, opt_state_alpha = opt_alpha.update(
            gradient_alpha,
            opt_state_alpha,
            params=eqx.filter(alpha, eqx.is_inexact_array),
        )
        alpha = eqx.apply_updates(alpha, updates_alpha)

    norm_metrics = {
        "param_q_norm": optax.global_norm(eqx.filter(q_network, eqx.is_inexact_array)),
        "grad_q_norm": optax.global_norm(gradient_q),
        "updates_q_norm": optax.global_norm(updates_q),
        "param_pi_norm": optax.global_norm(
            eqx.filter(pi_network, eqx.is_inexact_array)
        ),
        "grad_pi_norm": optax.global_norm(gradient_pi),
        "updates_pi_norm": optax.global_norm(updates_pi),
    }

    return (
        q_network,
        pi_network,
        alpha,
        q_target,
        opt_state_q,
        opt_state_pi,
        opt_state_alpha,
        {**outputs_q, **outputs_pi, **outputs_alpha, **norm_metrics},
    )
