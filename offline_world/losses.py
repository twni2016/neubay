import jax
import jax.numpy as jnp
import equinox as eqx


@eqx.filter_jit
def get_nll_and_unc(
    model,
    x: jnp.ndarray,  # (B, input_size)
    y: jnp.ndarray,  # (B, output_size)
) -> jnp.ndarray:
    (mu, sigma), unc = model.forward_same_data(x)  # (N, B, output_size)
    y_expanded = jnp.expand_dims(y, 0)  # add ensemble axis
    nll = (
        (
            0.5 * jnp.log(2 * jnp.pi)
            + jnp.log(sigma)
            + 0.5 * jnp.square((y_expanded - mu) / sigma)
        )
        .mean(-1)
        .mean(-1)
    )
    # here we average over output dimension to be dimension agnostic
    # return average nll, average unc, raw unc
    return nll, jax.tree.map(jnp.mean, unc), unc


@eqx.filter_jit
def get_mse_and_unc(
    model,
    x: jnp.ndarray,  # (B, input_size)
    y: jnp.ndarray,  # (B, output_size)
) -> jnp.ndarray:
    (mu, _), unc = model.forward_same_data(x)  # (N, B, output_size)
    y_expanded = jnp.expand_dims(y, 0)  # add ensemble axis
    mse = jnp.square(y_expanded - mu).mean(-1).mean(-1)
    # here we average over output dimension to be dimension agnostic
    # return average mse, average unc, raw unc
    return mse, jax.tree.map(jnp.mean, unc), unc


@eqx.filter_value_and_grad
def get_train_loss(
    model,
    x: jnp.ndarray,  # (N, B, input_size)
    y: jnp.ndarray,  # (N, B, output_size)
) -> jnp.ndarray:
    mu, sigma = model.forward_diff_data(x)  # (N, B, output_size)
    nll = (
        0.5 * jnp.log(2 * jnp.pi) + jnp.log(sigma) + 0.5 * jnp.square((y - mu) / sigma)
    )
    return nll.mean()  # scalar
