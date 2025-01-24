"""
Produces the 1D results of the broad comparison across architectures and
dynamics.

Full prediction learning with one-step supervised training.
"""

import os

DEFAULT_ROOT = os.path.dirname(os.path.abspath(__file__))
CONFIG_PREFIX = "configs/broad_comparison_1d"
TRAIN_DIR_ROOT = f"{DEFAULT_ROOT}/{CONFIG_PREFIX}/train"
VALID_DIR_ROOT = f"{DEFAULT_ROOT}/{CONFIG_PREFIX}/validation"

CONFIGS = [
    {
        "scenario_name": scene,
        "network_config": net
    }
    # for s in [0, 10, 20, 30, 40]
    for net in [
        "Conv;34;10;relu",  # 31'757 params, 11 receptive field per direction
        # "UNet;12;2;relu",  # 27'193 params, 29 receptive field per direction
        # "Res;26;8;relu",  # 32'943 params, 16 receptive field per direction
        "FNO;12;18;4;gelu",  # 32'527 params, inf receptive field per direction
        # "Dil;2;32;2;relu",  # 31'777 params, 20 receptive field per direction
    ]
    for scene in [
        # "diff_disp",
        "diff_burgers",
        "diff_kdv",
        # "diff_ks_cons",  # was not used in the paper in the end
        "diff_ks",
    ]
]


def for_training(ape_config, sampler_suffix):
    s = ape_config["scenario_name"]
    n = ape_config["network_config"].replace(";", "_")
    validation_hierarchy = f"{VALID_DIR_ROOT}/{s}"
    validation_dir = \
        f"{validation_hierarchy}/VALIDATION_OUT/trajectories"  # noqa
    suffix = f"{s}_{n}_{sampler_suffix}"
    hierarchy = f"{TRAIN_DIR_ROOT}/{s}/{n}"
    os.makedirs(hierarchy, exist_ok=True)
    output_dir = f"STUDY_OUT_{suffix}"
    output_config_file = f"{hierarchy}/config_{suffix}.json"

    return output_config_file, output_dir, validation_dir


def for_validation(ape_config):
    s = ape_config["scenario_name"]
    hierarchy = f"{VALID_DIR_ROOT}/{s}"
    os.makedirs(hierarchy, exist_ok=True)
    output_dir = "VALIDATION_OUT"
    output_config_file = f"{hierarchy}/config_{s}.json"

    return output_config_file, output_dir
