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
