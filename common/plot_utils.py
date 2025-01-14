import io

import numpy as np
import jax.numpy as jnp
from PIL import Image
import matplotlib.pyplot as plt
import exponax as ex


plt.style.use('ggplot')


def plot_grid(u, *, make_grid_args={}, plot_args={}):

    full_grid = ex.make_grid(
        **make_grid_args,
        full=True
    )
    plt.plot(full_grid[0], ex.wrap_bc(u)[0], **plot_args)


def mpl_to_tensorboard_image():

    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight')
    buf.seek(0)
    image = Image.open(buf).convert("RGB")
    rgb_image = np.asarray(image)
    buf.close()

    return rgb_image


def create_subplot_1d(nrows,
                      ncols,
                      domain_extent,
                      sim_ids,
                      tids,
                      meshes):

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10), sharex=True, sharey=True)
    labels = ["u_prev", "u_next", "u_next_hat"]
    u_prev, u_next, u_next_hat = meshes
    assert len(u_prev.shape) == 4
    l, u = jnp.min(jnp.asarray(meshes)) - 0.1, jnp.max(jnp.asarray(meshes)) + 0.1
    handles = []
    for row in range(nrows):
        for col in range(ncols):
            ax = axes[row, col]
            for label, state in zip(labels, [
                u_prev[row, col],
                u_next[row, col],
                u_next_hat[row, col]
            ]):
                p = ex.viz.plot_state_1d(
                    state,
                    domain_extent=domain_extent,
                    ax=ax,
                    xlabel="",
                    ylabel="",
                    vlim=(l, u),
                    alpha=0.7,
                    linestyle="--" if label == "u_next_hat" else "-"
                )
                if row == 0 and col == 0:
                    handles.append(p[0])
            ax.set_title(f"sim={sim_ids[row][0]} t={tids[col]}")

    fig.text(0.5, -0.02, "Space", ha="center", fontsize=15)
    fig.text(-0.02, 0.5, "Value", va="center", rotation="vertical", fontsize=15)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.draw()

    return mpl_to_tensorboard_image()


def scatter_plot(x,
                 y,
                 color_data=None,
                 cmap='Reds',
                 title='',
                 xlabel='p1',
                 ylabel='p2',
                 status=None,
                 color_map=None,
                 show_colorbar=False,
                 colorbar_label=''):

    plt.figure(figsize=(10, 8))

    if color_data is not None:
        sc = plt.scatter(x, y, c=color_data, cmap=cmap, edgecolor='k')
        if show_colorbar:
            plt.colorbar(sc, label=colorbar_label)
    elif status is not None:
        colors = np.where(status, 'red', 'blue')
        sc = plt.scatter(x, y, c=colors, alpha=0.7, edgecolor='k')
    else:
        sc = plt.scatter(x, y, edgecolor='k')

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.grid(True)
    plt.tight_layout()

    return mpl_to_tensorboard_image()

def delta_loss_scatter_plot(n, x, y, delta_loss):
    title = f"delta losses for sliding window (last) = {n} simulations"
    return scatter_plot(x, y, color_data=delta_loss, cmap='Reds', title=title, show_colorbar=True, colorbar_label='delta loss')


def bred_scatter_plot(x, y, status):
    title = "breed status"
    return scatter_plot(x, y, status=status, title=title)


def validation_loss_scatter_plot_by_sim(x, y, loss):
    title = f"validation loss for {len(loss)} simulations"
    return scatter_plot(x, y, color_data=loss, cmap='GnBu', title=title, show_colorbar=True, colorbar_label='loss')

