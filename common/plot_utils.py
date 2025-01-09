import math
import io
import tensorflow as tf
import matplotlib.pyplot as plt
import jax.numpy as jnp
import exponax as ex


def plt2tb():
    plt.tight_layout()
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    plt.close()
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=3)
    return image.numpy()


def plot_grid(u, t, **make_grid_args):

    full_grid = ex.make_grid(
        **make_grid_args,
        full=True
    )
    plt.plot(full_grid[0], ex.wrap_bc(u)[0], label=f"{t}th step")


def plot_u_states_1d(u_prev, u_next, u_next_hat, domain_extent=None, ax=None):
    """Helper function to plot u_prev, u_next, and u_next_hat using plot_state_1d."""
    
    state = jnp.stack([u_prev, u_next, u_next_hat], axis=0)
    print(state.shape) 
    labels = ["u_prev", "u_next", "u_next_hat"]
    
    ex.viz.plot_state_1d(
        state,
        vlim=(-1.0, 1.0),
        domain_extent=domain_extent,
        labels=labels,
        ax=ax,
        xlabel="Space",
        ylabel="Value"
    )

    return ax


def create_subplot_nx5(title, n, meshes, **kwargs):
    ncols = 5
    nrows = math.ceil(n / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(20, 4 * nrows), constrained_layout=True)
    fig.suptitle(title, fontsize=16)
    if nrows == 1:
        axes = axes[jnp.newaxis, :]
    
    for idx, (up, un, unh) in enumerate(meshes):
        row, col = divmod(idx, 5)
        ax = axes[row, col]
        plot_u_states_1d(up, un, unh, ax=ax, **kwargs)

    return plt2tb()
