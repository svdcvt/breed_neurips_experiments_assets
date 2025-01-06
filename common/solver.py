#! /usr/bin/python3

import os
import time
import argparse

import rapidjson
import matplotlib.pyplot as plt
import exponax as ex
from mpi4py import MPI
import numpy as np
import jax.numpy as jnp

from melissa_api import (  # type: ignore
    melissa_init,
    melissa_send,
    melissa_finalize
)
from melissa.launcher.schema import CONFIG_PARSE_MODE  # type: ignore

from scenarios import SCENARIOS, MelissaSpecificScenario

try:
    with open(os.getenv("CONFIG_FILE")) as json_file:
        CONFIG_DICT = rapidjson.load(json_file, parse_mode=CONFIG_PARSE_MODE)
except Exception as e:
    print(str(e))
    print("Please set CONFIG_FILE with configuration file path.")

VALIDATION_DIR = "trajectories"
VALDIATION_INPUT_PARAM_FILE = f"{VALIDATION_DIR}/input_parameters.npy"
SCENARIO_CONFIG = CONFIG_DICT["study_options"]["scenario_config"]
NB_STEPS = CONFIG_DICT["study_options"]["nb_time_steps"]
FIELD_PREV_POSITION = "preposition"
FIELD_POSITION = "position"


def plot_grid(u, t, **make_grid_args):

    full_grid = ex.make_grid(
        **make_grid_args,
        full=True
    )
    plt.plot(full_grid[0], ex.wrap_bc(u)[0], label=f"{t}th step")


def get_default_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--scenario",
        type=str,
        required=True,
        choices=list(SCENARIOS.keys()),
        help="Load the solver with this APEBench scenario"
    )
    parser.add_argument(
        "--ic-config",
        type=str,
        required=True,
        help="Load the solver with this APEBench IC generator"
    )
    parser.add_argument(
        "--store",
        action="store_true",
        help="Set this when you want to store the trajectories"
    )
    return parser


def online(stepper, ic, flattened_mesh_size):
    comm = MPI.COMM_WORLD

    melissa_init(FIELD_PREV_POSITION, flattened_mesh_size, comm)
    melissa_init(FIELD_POSITION, flattened_mesh_size, comm)

    u = ic
    st = time.time()
    for t in range(NB_STEPS):
        u_next = stepper(u)
        melissa_send(
            FIELD_PREV_POSITION,
            np.asarray(u).flatten()
        )
        melissa_send(
            FIELD_POSITION,
            np.asarray(u_next).flatten()
        )
        print(f"t={t} solved")

    melissa_finalize()
    print(f"Total time taken {time.time() - st:.2f} sec.")


def offline(stepper, ic):
    rollout_advection_stepper = ex.rollout(
        stepper,
        NB_STEPS,
        include_init=True
    )
    st = time.time()
    trajectory = rollout_advection_stepper(ic)
    sim_id = os.getenv("MELISSA_SIMU_ID")
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    jnp.save(f"{VALIDATION_DIR}/sim{sim_id}.npy", trajectory)
    print(f"Total time taken {time.time() - st:.2f} sec.")


def run_solver(scenario_name, store, sampled_ic_config):
    scenario = MelissaSpecificScenario(scenario_name, SCENARIO_CONFIG)
    stepper = scenario.get_stepper()
    data_shape = scenario.get_shape()
    flattened_mesh_size = np.prod(data_shape)
    ic = scenario.get_ic(sampled_ic_config)

    if store:
        offline(stepper, ic)
    else:
        online(stepper, ic, flattened_mesh_size)


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    run_solver(args.scenario, args.store, args.ic_config)
