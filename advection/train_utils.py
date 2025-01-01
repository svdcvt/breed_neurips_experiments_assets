import jax
import jax.numpy as jnp
import equinox as eqx


def normalize():
    pass


def denormalize():
    pass


def loss_fn(model, x, y):
    y_pred = jax.vmap(model)(x)
    mse = jnp.mean(jnp.square(y_pred - y))
    return mse


@eqx.filter_jit
def update_fn(model, optimizer, x, y, opt_state=None):
    if opt_state is None:
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    loss, grad = eqx.filter_value_and_grad(loss_fn)(model, x, y)
    updates, new_state = optimizer.update(grad, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return new_model, new_state, loss
