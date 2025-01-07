import matplotlib.pyplot as plt
import exponax as ex


def plot_grid(u, t, **make_grid_args):

    full_grid = ex.make_grid(
        **make_grid_args,
        full=True
    )
    plt.plot(full_grid[0], ex.wrap_bc(u)[0], label=f"{t}th step")
