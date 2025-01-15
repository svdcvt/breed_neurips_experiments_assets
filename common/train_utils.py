import torch
import jax
import jax.numpy as jnp
import equinox as eqx
import exponax as ex

done = False
DIM2AXIS = {
    1: (1,),
    2: (1, 2),
    3: (1, 2, 3)
}


def jax2torch(jax_array):
    numpy_array = jnp.asarray(jax_array).copy()
    return torch.tensor(numpy_array)


def normalize():
    pass


def denormalize():
    pass


def get_grads_stats(grads):

    grads_flat, _ = jax.tree_util.tree_flatten(eqx.filter(grads, eqx.is_array))
    flat_grads = [g.flatten() for g in grads_flat]
    grads_concat = jnp.concatenate(flat_grads)
    total_norm = 0.0
    for g in flat_grads:
        total_norm += jnp.linalg.norm(g, ord=2) ** 2
    total_norm = jnp.sqrt(total_norm)
    mean = jnp.mean(grads_concat)
    variance = jnp.var(grads_concat)

    return {
        "l2-norm": total_norm.item(),
        "mean": mean.item(),
        "var": variance.item()
    }


def loss_fn(model, x, y, is_valid=False):
    y_pred = jax.vmap(model)(x)
    mse_per_sample = jax.vmap(
        ex.metrics.nRMSE
    )(y_pred, y)
    batch_mse = jnp.mean(mse_per_sample)
    if is_valid:
        return batch_mse, mse_per_sample, y_pred
    return batch_mse, mse_per_sample


def rollout_loss_fn(model, x, n=5):
    """x is the batch of trajectories (batch/sim, tsteps, channel, *dims)"""
    ics = x[:, 0]
    y = x[:, 1:n+1]
    y_pred = jax.vmap(
        ex.rollout(
            model,
            n,
            include_init=False,
        )
    )(ics)

    mse_per_traj = jax.vmap(
        ex.metrics.nRMSE,
        in_axes=1
    )(y_pred, y)

    return jnp.mean(mse_per_traj), mse_per_traj, y_pred


@eqx.filter_jit
def update_fn(model, optimizer, x, y, opt_state=None):
    if opt_state is None:
        opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
    eval_grad = eqx.filter_value_and_grad(
        loss_fn,
        has_aux=True
    )
    (loss, loss_per_sample), grads = eval_grad(model, x, y)
    updates, new_state = optimizer.update(grads, opt_state, model)
    new_model = eqx.apply_updates(model, updates)
    return (
        new_model,
        new_state,
        loss,
        loss_per_sample,
        grads
    )
