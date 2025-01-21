import io

import numpy as np
import jax.numpy as jnp
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
import exponax as ex


# plt.style.use('ggplot')


def plot_grid(u, *, make_grid_args={}, plot_args={}):

    full_grid = ex.make_grid(
        **make_grid_args,
        full=True
    )
    plt.plot(full_grid[0], ex.wrap_bc(u)[0], **plot_args)


def plot_error(ax, u_next, u_next_hat):

    error = jnp.abs(u_next - u_next_hat)
    rmse = jnp.sqrt(np.mean((u_next - u_next_hat) ** 2))
    ax.imshow(error, cmap="Reds", vmin=0, vmax=None)
    ax.set_title(f"error rmse={rmse:.2e}")

    return ax


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

    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10),
                             sharex=True, sharey=True)
    labels = ["u_prev", "u_next", "u_next_hat"]
    u_prev, u_next, u_next_hat = meshes
    assert len(u_prev.shape) == 4
    llim = jnp.min(jnp.asarray(meshes)) - 0.1
    ulim = jnp.max(jnp.asarray(meshes)) + 0.1
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
                    vlim=(llim, ulim),
                    alpha=0.7,
                    linestyle="--" if label == "u_next_hat" else "-"
                )
                if row == 0 and col == 0:
                    handles.append(p[0])
            ax.set_title(f"sim={sim_ids[row][0]} t={tids[col]}")

    fig.text(0.5, -0.02, "Space", ha="center", fontsize=15)
    fig.text(-0.02, 0.5, "Value", va="center", rotation="vertical",
             fontsize=15)
    fig.legend(handles, labels, loc="upper center", ncol=len(labels),
               bbox_to_anchor=(0.5, 1.05))
    plt.tight_layout()
    plt.draw()

    return fig
    # return mpl_to_tensorboard_image()


def create_subplot_2d(nrows,
                      domain_extent,
                      sim_ids,
                      tids,
                      meshes):

    ncols = 4
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10),
                             sharex=True, sharey=True)
    labels = ["u_prev", "u_next", "u_next_hat", "error"]
    assert len(meshes[0].shape) == 4, \
        f"GOT shape of meshes[0] = {meshes[0].shape}"
    l, u = jnp.min(jnp.asarray(meshes)), jnp.max(jnp.asarray(meshes))
    for row in range(nrows):
        ax = axes[row]
        for col in range(ncols):
            ex.viz.plot_state_2d(
                state=meshes[col][row],
                domain_extent=domain_extent,
                ax=ax[col],
                vlim=(l, u),
            )
            ax[col].set_title(labels[col])
        # endfor
        row_title = f"sim={sim_ids[row]} tstep={tids[row]}"
        fig.text(
            0.05,
            1 - (row + 0.5) / nrows,
            row_title,
            ha='right',
            va='center',
            fontsize=12,
        )

        ax[-1] = plot_error(ax[-1], meshes[1][row], meshes[2][row])
    # endfor
    plt.tight_layout()
    plt.draw()

    return fig
    # return mpl_to_tensorboard_image()


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

    fig = plt.figure(figsize=(10, 8))

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

    return fig
    # return mpl_to_tensorboard_image()


def delta_loss_scatter_plot(n, x, y, delta_loss):
    title = f"delta losses for sliding window (last) = {n} simulations"
    return scatter_plot(
        x, y,
        color_data=delta_loss, cmap='Reds', title=title,
        show_colorbar=True, colorbar_label='delta loss'
    )


def bred_scatter_plot(x, y, status):
    title = "breed status"
    return scatter_plot(x, y, status=status, title=title)


def validation_loss_scatter_plot_by_sim(x, y, loss):
    title = f"validation loss for {len(loss)} simulations"
    return scatter_plot(
        x, y,
        color_data=loss, cmap='GnBu',
        title=title, show_colorbar=True, colorbar_label='loss'
    )


def plot_seen_count_histogram(seen_counts):
    fig = plt.figure()
    ax = plt.gca()
    ax.yaxis.set_major_locator(MaxNLocator(integer=True))
    plt.hist(np.array(seen_counts), edgecolor='k')
    plt.xlabel("Number of seen counts")
    plt.ylabel("Frequency")
    plt.tight_layout()
    
    return fig
    # return mpl_to_tensorboard_image()


class DynamicHistogram():
    def __init__(self, title='Histogram', cmap='magma',
                 bins=50, size=(8, 5), show_last=0):
        self.cmap = plt.get_cmap(cmap)
        self.bins = bins
        self.fig, self.axes = plt.subplots(1, 1, figsize=size)
        self.axes.set_title(title)
        self.axes.set_xlabel('Value')
        self.axes.set_ylabel('Log-Density')
        self.first = 0
        self.current = 0
        self.show_last = show_last
        self.cbar = self.fig.colorbar(
            mpl.cm.ScalarMappable(cmap=self.cmap),
            ax=self.axes, orientation='vertical', label='Iteration')

    def add_histogram_step(self, data):
        # 1. update counter
        self.current += 1
        # 2. add histogram
        self.axes.hist(data, bins=self.bins, histtype='stepfilled', log=True)
        # 3. update colors
        for i, color in enumerate(map(self.cmap, np.linspace(
            0, 1, self.current - self.first
        ))):
            self.axes.patches[i].set_facecolor(color)
        # 4. update xlim
        xmin, xmax = min(data), max(data)
        if (
            (self.axes.get_xlim()[0] / xmin) <= 0.5
            or xmin < self.axes.get_xlim()[0]
        ):
            self.axes.set_xlim(xmin, None)
        if (
            (xmax / self.axes.get_xlim()[1]) <= 0.5
            or xmax > self.axes.get_xlim()[1]
        ):
            self.axes.set_xlim(None, xmax)
        # 5. when theres enough artists - remove first
        if self.show_last != 0 and self.current > self.show_last:
            self.axes.patches[0].remove()
            self.first = self.current - self.show_last
        # 6. update colorbar limit
        self.cbar.mappable.set_norm(
            mpl.colors.Normalize(self.first, self.current)
        )
