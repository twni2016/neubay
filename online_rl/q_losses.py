import jax
from jax import numpy as jnp
import equinox as eqx
import optax
from online_rl.common import soft_update, huber


def ddqn_loss(q_network, q_target, tape, gamma, key):
    """
    The tape loss function is same as Markov loss, but with the following changes:
    1. We have initial states
    2. We have a mask for the start of each episode to extract next_q and current_q
    """
    initial_state = q_network.initial_state()  # L[(1, d_hidden)] complex

    B = tape["next_reward"].shape[0]
    batch_idx = jnp.arange(B)

    q_values = q_network(
        tape["observation"], initial_state, tape["start"], key=key
    )  # input: (T, d_input), q_values: (T, d_action)
    selected_q = q_values[batch_idx, tape["action"]][:-1]  # (T-1)

    next_q_action_idx = jax.lax.stop_gradient(q_values).argmax(-1).flatten()
    next_q = jax.lax.stop_gradient(
        q_target(
            tape["observation"],
            initial_state,
            tape["start"],
            key=key,
        )
    )
    next_q = next_q[batch_idx, next_q_action_idx][1:]  # (T-1)

    target = (
        tape["next_reward"][:-1] + (1.0 - tape["next_terminated"][:-1]) * gamma * next_q
    )
    error = selected_q - target
    loss = huber(error)

    # because the end of the last episode is right before the start of the next episode
    mask = 1.0 - tape["start"][1:]  # (T-1)
    loss = (loss * mask).sum() / mask.sum()

    return loss, {
        "loss": loss,
        "q_mean": (selected_q * mask).sum() / mask.sum(),
        "target_mean": (target * mask).sum() / mask.sum(),
        "target_network_mean": (next_q * mask).sum() / mask.sum(),
    }


@eqx.filter_jit
def ddqn_update(q_network, q_target, data, opt, opt_state, gamma, tau, loss_key):
    (loss, outputs_q), gradient = eqx.filter_value_and_grad(ddqn_loss, has_aux=True)(
        q_network, q_target, data, gamma, loss_key
    )
    updates, opt_state = opt.update(
        gradient, opt_state, params=eqx.filter(q_network, eqx.is_inexact_array)
    )
    q_network = eqx.apply_updates(q_network, updates)
    q_target = soft_update(q_network, q_target, tau=tau)

    norm_metrics = {
        "param_norm": optax.global_norm(eqx.filter(q_network, eqx.is_inexact_array)),
        "grad_norm": optax.global_norm(gradient),
        "updates_norm": optax.global_norm(updates),
    }

    return (
        q_network,
        q_target,
        opt_state,
        {**outputs_q, **norm_metrics},
    )
