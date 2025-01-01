#! /usr/bin/env python3

import time
import rapidjson
import argparse

from mpi4py import MPI
import exponax as ex
import numpy as np
import matplotlib.pyplot as plt

from melissa_api import (  # type: ignore
    melissa_init,
    melissa_send,
    melissa_finalize
)

from melissa.launcher.schema import CONFIG_PARSE_MODE  # type: ignore


with open("config_slurm.json") as json_file:
    config_dict = rapidjson.load(json_file, parse_mode=CONFIG_PARSE_MODE)


NB_STEPS = config_dict["study_options"]["nb_time_steps"]
NB_DIMS = config_dict["study_options"]["advection"]["nb_dims"]
DOMAIN_EXTENT = config_dict["study_options"]["advection"]["domain_extent"]
NUM_POINTS = config_dict["study_options"]["advection"]["nb_points"]
DT = config_dict["study_options"]["advection"]["dt"]
VELOCITY = config_dict["study_options"]["advection"]["velocity"]


def plot_grid(u, t):

    full_grid = ex.make_grid(
        NB_DIMS,
        DOMAIN_EXTENT,
        NUM_POINTS,
        full=True
    )
    plt.plot(full_grid[0], ex.wrap_bc(u)[0], label=f"{t}th step")


def advection_solver(**kwargs):

    make_ic = ex.ic.SineWaves1d(
        DOMAIN_EXTENT,
        (kwargs["amplitude"],),
        (1,),
        (kwargs["phase"],)
    )
    print(kwargs)
    advection_stepper = ex.stepper.Advection(
        num_spatial_dims=NB_DIMS,
        domain_extent=DOMAIN_EXTENT,
        num_points=NUM_POINTS,
        dt=DT,
        velocity=VELOCITY
    )
    print(advection_stepper)
    ic = make_ic(
        ex.make_grid(
            NB_DIMS,
            DOMAIN_EXTENT,
            NUM_POINTS
        ),
    )

    comm = MPI.COMM_WORLD
    field_previous_position = "preposition"
    field_position = "position"
    melissa_init(field_previous_position, NUM_POINTS, comm)
    melissa_init(field_position, NUM_POINTS, comm)

    u = ic
    st = time.time()
    for t in range(NB_STEPS):
        u_next = advection_stepper(u)
        melissa_send(
            field_previous_position,
            np.asarray(u).flatten()
        )
        melissa_send(
            field_position,
            np.asarray(u_next).flatten()
        )
        print(f"t={t} solved")

    melissa_finalize()

    print(f"Total time taken {time.time() - st:.2f} sec.")


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--amplitude",
        type=float,
        required=True,
        help="IC Amplitude of the sine wave."
    )
    parser.add_argument(
        "--phase",
        type=float,
        required=True,
        help="IC Phase for the sine wave."
    )

    return parser.parse_args()


if __name__ == "__main__":

    args = parse_arguments()
    advection_solver(**vars(args))
