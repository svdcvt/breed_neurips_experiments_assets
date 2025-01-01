#! /usr/bin/env python3

import argparse

from mpi4py import MPI
import exponax as ex
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt

from melissa_api import (  # type: ignore
    melissa_init,
    melissa_send,
    melissa_finalize
)


def plot_grid(u, t, **kwargs):

    del kwargs["dt"], kwargs["velocity"]

    full_grid = ex.make_grid(
        1,
        kwargs["domain_extent"],
        kwargs["num_points"],
        full=True
    )
    plt.plot(full_grid[0], ex.wrap_bc(u)[0], label=f"{t}th step")


def advection1d_solver(tsteps, **kwargs):

    def make_ic(x):
        return jnp.sin(2 * jnp.pi * x / kwargs["domain_extent"])

    print(kwargs)
    advection_stepper = ex.stepper.Advection(
        num_spatial_dims=1,
        **kwargs
    )
    print(advection_stepper)
    ic = make_ic(
        ex.make_grid(
            1,
            kwargs["domain_extent"],
            kwargs["num_points"]
        )
    )

    comm = MPI.COMM_WORLD
    field_previous_position = "preposition"
    field_position = "position"
    melissa_init(field_previous_position, kwargs["num_points"], comm)
    melissa_init(field_position, kwargs["num_points"], comm)

    u = ic
    for t in range(tsteps):
        u_next = advection_stepper(u)
        melissa_send(field_previous_position, np.asarray(u))
        melissa_send(field_position, np.asarray(u_next))
        print(f"t={t} solved")

    melissa_finalize()

    plot_grid(ic, 0, **kwargs)
    plot_grid(u_next, t, **kwargs)
    plt.legend()
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--tsteps",
        type=int,
        default=100,
        help="Number of timesteps to solve."
    )
    parser.add_argument(
        "--domain-extent",
        type=float,
        default=1.0,
        help="The domain of the simulation."
    )
    parser.add_argument(
        "--num-points",
        type=int,
        default=100,
        help="Number of points in the simulation grid."
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.1,
        help="Time step size for the simulation."
    )
    parser.add_argument(
        "--velocity",
        type=float,
        required=True,
        help="Velocity parameter for the simulation."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    advection1d_solver(**vars(args))
