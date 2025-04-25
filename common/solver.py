#! /usr/bin/env python3

import os
import sys
import time
import argparse

import rapidjson
import exponax as ex
from mpi4py import MPI
import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.colors import CenteredNorm
from matplotlib.gridspec import GridSpec

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
        "--offline",
        action="store_true",
        help="Set this when you want to store the trajectories"
    )

    return parser


def run_online(stepper, u, flattened_mesh_size, sampled_ic_config):
    if np.random.rand() < 0.01:
        print('SUPPOSE TO HAVE A PLOT at', os.getcwd())
        to_plot = True
        traj = np.empty((NB_STEPS+1, flattened_mesh_size))
    else:
        to_plot = False
    comm = MPI.COMM_WORLD

    melissa_init(FIELD_PREV_POSITION, flattened_mesh_size, comm)
    melissa_init(FIELD_POSITION, flattened_mesh_size, comm)

    if jnp.isnan(u).any():
        print("IC config:", sampled_ic_config, file=sys.stderr)
        print(
            f"NaN values encountered in IC. Aborting the solver.",
            file=sys.stderr
        )
        os._exit(1)
    if to_plot:
        traj[0] = np.asarray(u).flatten()
    st = time.time()
    for t in range(NB_STEPS):
        melissa_send(
            FIELD_PREV_POSITION,
            np.asarray(u).flatten()
        )
        u = stepper(u)
        
        if jnp.isnan(u).any():
            print(
                f"NaN values encountered at t={t}. Aborting the solver.",
                file=sys.stderr
            )
            print('Previous step stats:', u.min(), u.max(), u.mean(), file=sys.stderr)
            print("IC config:", sampled_ic_config, file=sys.stderr)
            os._exit(1)

        melissa_send(
            FIELD_POSITION,
            np.asarray(u).flatten()
        )
        print(f"t={t} solved")
        # u = u_next.copy()
        if to_plot:
            traj[t+1] = np.asarray(u).flatten()

    melissa_finalize()
    print(f"Total time taken {time.time() - st:.2f} sec.")
    if to_plot:
        pass

def run_offline(stepper, ic, sampled_ic_config):
    rollout_stepper = ex.rollout(
        stepper,
        NB_STEPS,
        include_init=True
    )
    st = time.time()
    trajectory = rollout_stepper(ic)
    if jnp.isnan(trajectory).any():
        print(
            f"NaN values encountered in the trajectory.",
            file=sys.stderr
        )
    sim_id = os.getenv("MELISSA_SIMU_ID")
    os.makedirs(VALIDATION_DIR, exist_ok=True)
    jnp.save(f"{VALIDATION_DIR}/sim{sim_id}.npy", trajectory)
    print(f"Total time taken {time.time() - st:.2f} sec.")
    print("Trajectory shape:", trajectory.shape)
    print("Trajectory min, max, mean:", trajectory.min(), trajectory.max(), trajectory.mean())
    print("Trajectory std:", trajectory.std())
    
    if np.random.rand() < 0.1:
        traj = np.array(trajectory).squeeze(1)

        fig = plt.figure(layout="constrained", figsize=(15,9))
        gs = GridSpec(3, 5, figure=fig)
        ax0 = fig.add_subplot(gs[0, :3])
        ax1 = fig.add_subplot(gs[0, 3:])
        ax2 = fig.add_subplot(gs[1:, :])
        ax = [ax0, ax1, ax2]
        
        # fig, ax = plt.subplots(1, 3, )
        ic_pars = [f'{float(x):.3f}' for x in sampled_ic_config.split(';')[1:-2]]

        fig.suptitle('Amps:' + ', '.join(ic_pars[::2]) + '\nPhs:' + ', '.join(ic_pars[1::2]), fontsize=10)
        ax[0].plot(traj[0], linewidth=1, label='IC', color='darkgreen')
        ax[0].plot(traj[-1], linewidth=1,label='Last', color='darkred')
        ax[0].plot(traj[1], linewidth=1, linestyle='--',label='IC+1', color='limegreen')
        ax[0].plot(traj[-2], linewidth=1,linestyle='--',label='Last-1', color='salmon')
        ax[0].legend(loc='upper right', fontsize=6)
        
        im = ax[1].imshow(traj.T, cmap='coolwarm', norm=CenteredNorm(), aspect='auto')
        fig.colorbar(im, ax=ax[1])
        ax[1].set_title('Trajectory')
        ax[1].set_xlabel('Time step')
        ax[1].set_ylabel('Mesh points')
        cmap = plt.get_cmap('RdYlGn_r')
        for i in range(traj.shape[0])[::10]:
            ax[2].plot(traj[i],c=cmap(i/100), linewidth=0.5 if i % 100 else 1.5, alpha=0.7)
        ax[2].axis('off')

        trajectory_std = np.std(traj, axis=-1).mean()
        max_abs = np.max(np.abs(traj))

        plot_dir = os.path.join(VALIDATION_DIR, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        fig.savefig(f'{plot_dir}/traj_std_{trajectory_std:3.1e}_maxabs_{max_abs:3.1e}_id_{sim_id}.png', dpi=200)
        plt.close(fig)
        print("FINISHED!")


def run_solver(offline, sampled_ic_config):
    scenario = MelissaSpecificScenario(
        sampled_ic_config=sampled_ic_config,
        **SCENARIO_CONFIG
    )
    stepper = scenario.get_stepper()
    print(stepper)
    try:
        print("difficulty gamma and delta from object")
        print(stepper.linear_difficulties)
        print(stepper.convection_difficulty)
    except:
        pass
    try:
        print('normalized_linear_coefficients and normalized_convection_scale')
        print(stepper.normalized_linear_coefficients)
        print(stepper.normalized_convection_scale)
        print("difficulty gamma and delta")
        difficulty_linear_coefficients = [
            alpha * stepper.num_points**j * 2 ** (j - 1) * 1
            for j, alpha in enumerate(stepper.normalized_linear_coefficients)
        ]
        difficulty_convection_scale = stepper.normalized_convection_scale * (
            1.0 * stepper.num_points * 1
        )
        print(difficulty_linear_coefficients)
        print(difficulty_convection_scale)
    except:
        pass
    data_shape = scenario.get_shape()
    flattened_mesh_size = np.prod(data_shape)
    input_fn_config = SCENARIO_CONFIG.get("input_fn_config", {})
    ic = scenario.get_ic_mesh(**input_fn_config)

    if offline:
        run_offline(stepper, ic, sampled_ic_config)
    else:
        run_online(stepper, ic, flattened_mesh_size, sampled_ic_config)


if __name__ == "__main__":
    parser = get_default_parser()
    args = parser.parse_args()
    run_solver(args.offline, args.ic_config)
