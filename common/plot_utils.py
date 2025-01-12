import io

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import exponax as ex


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

    fig, axes = plt.subplots(nrows, ncols, figsize=(10, 10))

    for row in range(nrows):
        u_prev, u_next, u_next_hat = meshes[row]
        for col in range(ncols):
            ax = axes[row, col]
            for (label, state) in [
                ("u_prev", u_prev[col]),
                ("u_next", u_next[col]),
                ("u_next_hat", u_next_hat[col])
            ]:
                ex.viz.plot_state_1d(
                    state,
                    domain_extent=domain_extent,
                    labels=[label],
                    ax=ax,
                    alpha=0.7,
                    linestyle="--" if label == "u_next_hat" else "-"
                )
                ax.set_title(f"{label} sim={sim_ids[row]} t={tids[col]}")

    plt.draw()

    return mpl_to_tensorboard_image()
