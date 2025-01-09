#! /usr/bin/env python3

import os
import time
import argparse

import rapidjson
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

from scenarios import MelissaSpecificScenario
from constants import (
    FIELD_PREV_POSITION,
    FIELD_POSITION,
    VALIDATION_DIR
)

try:
    with open(os.getenv("CONFIG_FILE")) as json_file:
        CONFIG_DICT = rapidjson.load(json_file, parse_mode=CONFIG_PARSE_MODE)
except Exception as e:
    print(str(e))
    print(
        "Please set CONFIG_FILE with configuration file path",
        "to load the scenario configuration for the solver."
    )
    raise Exception from e

SCENARIO_CONFIG = CONFIG_DICT["study_options"]["scenario_config"]
NB_STEPS = CONFIG_DICT["study_options"]["nb_time_steps"]


def get_default_parser():
    parser = argparse.ArgumentParser()
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
        u = u_next.copy()

    melissa_finalize()
    print(f"Total time taken {time.time() - st:.2f} sec.")


def offline(stepper, ic):
    rollout_stepper = ex.rollout(
        stepper,
        NB_STEPS,
        include_init=True
    )
    st = time.time()
    trajectory = rollout_stepper(ic)
    sim_id = os.getenv("MELISSA_SIMU_ID")
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    jnp.save(f"{VALIDATION_DIR}/sim{sim_id}.npy", trajectory)
    print(f"Total time taken {time.time() - st:.2f} sec.")


def run_solver(store, sampled_ic_config):
    scenario = MelissaSpecificScenario(
        sampled_ic_config=sampled_ic_config,
        **SCENARIO_CONFIG
    )
    stepper_config = SCENARIO_CONFIG.get("stepper_config", {})
    stepper = scenario.get_stepper(**stepper_config)
    data_shape = scenario.get_shape()
    flattened_mesh_size = np.prod(data_shape)
    input_fn_config = SCENARIO_CONFIG.get("input_fn_config", {})
    ic = scenario.get_ic_mesh(**input_fn_config)

    if store:
        offline(stepper, ic)
    else:
        online(stepper, ic, flattened_mesh_size)


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    # the following can be class-specific default arguments
    # to be overriden
    run_solver(args.store, args.ic_config)
