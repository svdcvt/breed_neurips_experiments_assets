"""
Trains nonlinear emulators for the (linear) 1D advection equation under varying
difficulty in terms of the `advction_gamma` (=CFL).
"""

import os

DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PREFIX = "configs/advection_1d_varying_diff"
TRAIN_DIR_ROOT = f"{DEFAULT_ROOT}/{CONFIG_PREFIX}/train"
VALID_DIR_ROOT = f"{DEFAULT_ROOT}/{CONFIG_PREFIX}/validation"


CONFIGS = [
    {
        "scenario_name": "diff_adv",
        "network_config": net,
        "advection_gamma": advection_gamma,
    }
    for net in [
        *[f"Conv;34;{depth};relu" for depth in [0]],  # , 1, 2, 10]],
        # "UNet;12;2;relu",  # 27'193 params, 29 receptive field per direction
        # "Res;26;8;relu",  # 32'943 params, 16 receptive field per direction
        "FNO;12;18;4;gelu",  # 32'527 params, inf receptive field per direction
        # "Dil;2;32;2;relu",  # 31'777 params, 20 receptive field per direction
    ]
    for advection_gamma in [
        0.5,
        2.5,
        10.5,
    ]
]


def for_training(ape_config, sampler_suffix):
    s = ape_config["scenario_name"]
    n = ape_config["network_config"].replace(";", "_")
    g = str(ape_config["advection_gamma"]).replace(".", "_")
    validation_hierarchy = f"{VALID_DIR_ROOT}/{s}/gamma_{g}"
    validation_suffix = f"{s}_{g}"
    validation_dir = \
        f"{validation_hierarchy}/VALIDATION_OUT_{validation_suffix}/trajectories"  # noqa
    suffix = f"{s}_{n}_{g}_{sampler_suffix}"
    hierarchy = f"{TRAIN_DIR_ROOT}/{s}/{n}/gamma_{g}"
    os.makedirs(hierarchy, exist_ok=True)
    output_dir = f"STUDY_OUT_{suffix}"
    output_config_file = f"{hierarchy}/config_{suffix}.json"

    return output_config_file, output_dir, validation_dir


def for_validation(ape_config):
    s = ape_config["scenario_name"]
    g = str(ape_config["advection_gamma"]).replace(".", "_")
    suffix = f"{s}_{g}"
    hierarchy = f"{VALID_DIR_ROOT}/{s}/gamma_{g}"
    os.makedirs(hierarchy, exist_ok=True)
    output_dir = f"VALIDATION_OUT_{suffix}"
    output_config_file = f"{hierarchy}/config_{suffix}.json"

    return output_config_file, output_dir
