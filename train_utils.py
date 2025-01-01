import jax
import jax.numpy as jnp
import equinox as eqx
import pdequinox as pdeqx
import optax


def get_mlp(dl_config):
    return pdeqx.arch.MLP(
        1, 1, 1,
        num_points=dl_config.get("nb_points", 100),
        width_size=dl_config.get("width_size", 64),
        depth=dl_config.get("depth", 3),
        boundary_mode="dirichlet",
        key=jax.random.PRNGKey(0)
    )


def get_optimizer(dl_config):
    return optax.adam(dl_config.get("lr", 3e-4))


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
