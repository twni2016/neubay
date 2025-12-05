import jax
from jax import numpy as jnp
from jax import random
import equinox as eqx
import optax
from online_rl.common import soft_update, huber
from online_rl.ac_modules import select_critics


"""
Recurrent SAC loss
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
    """
    The tape loss function is same as Markov loss, but with the following changes:
    1. We have initial states
    2. We have a mask for the start of each episode to extract next_q and current_q
    3. We have real masks to balance the real and imaginary data
    """
    keys = random.split(key, 5)

    ## first get Q(h, a) of shape (N,T-1), where h is (T, d_input)
    qs = q_network(
        tape["observation"],
        q_network.initial_state(),
        tape["start"],
        tape["action"],
        key=keys[0],
    )[:, :-1]

    ## then get Q_tar(h', a') - log pi(a' | h') where a' ~ pi(h')
    next_action_dist = pi_network(
        tape["observation"], pi_network.initial_state(), tape["start"], key=keys[1]
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
        tape["observation"],
        sampled_q_target.initial_state(),
        tape["start"],
        next_action,
        key=keys[4],
    )[:, 1:]
    next_q = jnp.min(next_qs, axis=0)  # (M,T-1) -> (T-1)

    if backup_entropy:
        next_q = next_q - alpha() * next_action_logp[1:].sum(-1)
    target = jax.lax.stop_gradient(
        tape["next_reward"][:-1] + (1.0 - tape["next_terminated"][:-1]) * gamma * next_q
    )

    q_diff = qs - jnp.expand_dims(target, 0)
    loss = huber(q_diff).mean(0)  # (T-1)

    # because the end of the last episode is right before the start of the next episode
    mask = 1.0 - tape["start"][1:]  # (T-1)

    if "real" in tape:  # mix of real and imaginary data
        real_mask = mask * tape["real"][:-1]  # (T-1)
        imag_mask = mask * (1.0 - tape["real"][:-1])  # (T-1)
        real_sum = jnp.clip(real_mask.sum(), min=1)
        imag_sum = jnp.clip(imag_mask.sum(), min=1)
        real_loss = (loss * real_mask).sum() / real_sum
        imag_loss = (loss * imag_mask).sum() / imag_sum

        loss = real_weight * real_loss + (1.0 - real_weight) * imag_loss
        return loss, {
            "loss_q": loss,
            "loss_q_real": real_loss,
            "loss_q_imag": imag_loss,
            "q_mean": (qs[0] * mask).sum() / mask.sum(),
            "real_q_mean": (qs[0] * real_mask).sum() / real_sum,
            "imag_q_mean": (qs[0] * imag_mask).sum() / imag_sum,
        }

    else:  # pure data
        loss = (loss * mask).sum() / mask.sum()

        return loss, {
            "loss_q": loss,
            "q_mean": (qs[0] * mask).sum() / mask.sum(),
        }


def sac_pi_loss(pi_network, q_network, alpha, tape, real_weight, key):
    # log pi(a | h) - Q(h, a) where a ~ pi(z)
    keys = random.split(key, 3)
    action_dist = pi_network(
        tape["observation"], pi_network.initial_state(), tape["start"], key=keys[0]
    )
    action, action_logp = action_dist.sample_and_log_prob(seed=keys[1])

    qs = q_network(
        tape["observation"],
        q_network.initial_state(),
        tape["start"],
        action,
        key=keys[2],
    )[
        :, :-1
    ]  # (N, T-1)
    q = qs.mean(0)
    loss = alpha() * action_logp[:-1].sum(-1) - q
    entropy = -action_logp[:-1].sum(-1)

    mask = 1.0 - tape["start"][1:]  # (T-1)

    if "real" in tape:  # mix of real and imaginary data
        real_mask = mask * tape["real"][:-1]  # (T-1)
        imag_mask = mask * (1.0 - tape["real"][:-1])  # (T-1)
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
        loss = (loss * mask).sum() / mask.sum()
        return loss, {
            "entropy": jax.lax.stop_gradient(entropy),  # this is used for alpha loss
            "loss_pi": loss,
        }


def sac_alpha_loss(alpha, entropy, tape, real_weight):
    if alpha.target_entropy is None:
        loss = entropy  # a placeholder
    else:
        loss = alpha() * (entropy - alpha.target_entropy)

    mask = 1.0 - tape["start"][1:]  # (T-1)

    if "real" in tape:  # mix of real and imaginary data
        real_mask = mask * tape["real"][:-1]  # (T-1)
        imag_mask = mask * (1.0 - tape["real"][:-1])  # (T-1)
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
        loss = (loss * mask).sum() / mask.sum()
        return loss, {
            "alpha": jax.lax.stop_gradient(alpha()),
            "pi_entropy": (entropy * mask).sum() / mask.sum(),
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
