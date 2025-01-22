import os
import argparse
from apebench_study import CONFIGS as APEBENCH_STUDIES
from modify_json import modify_json

DEFAULT_ROOT = os.getenv(
    "APEBENCH_ROOT", os.path.dirname(os.path.abspath(__file__))
)

# override through command line options
TRAIN_CONFIGS_ROOT = f"{DEFAULT_ROOT}/configs/train"
VALID_CONFIGS_ROOT = f"{DEFAULT_ROOT}/configs/validation"
TRAIN_FILE = f"{DEFAULT_ROOT}/default_config_slurm.json"
VALID_FILE = f"{DEFAULT_ROOT}/default_config_offline.json"


def for_training(train_dir_root, valid_dir_root, input_train_file, ape_config):
    for sampler_type, sampler_suffix in [("uniform", "uniform"), ("breed", "breed"), ("breed", "breed_mix")]:

        s = ape_config["scenario_name"]
        n = ape_config["network_config"].replace(";", "_")
        g = str(ape_config["advection_gamma"]).replace(".", "_")
        validation_hierarchy = f"{valid_dir_root}/{s}/gamma_{g}"
        validation_suffix = f"{s}_{g}"
        validation_directory = \
            f"{validation_hierarchy}/VALIDATION_OUT_{validation_suffix}/trajectories"  # noqa
        suffix = f"{s}_{n}_{g}_{sampler_suffix}"
        hierarchy = f"{train_dir_root}/{s}/{n}/gamma_{g}"
        os.makedirs(hierarchy, exist_ok=True)
        output_dir = f"STUDY_OUT_{suffix}"
        output_config_file = f"{hierarchy}/config_{suffix}.json"

        modify_json(
            sampler=sampler_type,
            input_config_file=input_train_file,
            output_config_file=output_config_file,
            output_dir=output_dir,
            scenario_config=ape_config,
            validation_directory=validation_directory,
            use_true_mixing=sampler_suffix == "breed_mix"
        )
        print(os.path.split(output_config_file)[-1])


def for_validation(valid_dir_root, input_valid_file, ape_config):
    s = ape_config["scenario_name"]
    g = str(ape_config["advection_gamma"]).replace(".", "_")
    suffix = f"{s}_{g}"
    hierarchy = f"{valid_dir_root}/{s}/gamma_{g}"
    os.makedirs(hierarchy, exist_ok=True)
    output_dir = f"VALIDATION_OUT_{suffix}"
    output_config_file = f"{hierarchy}/config_{suffix}.json"
    modify_json(
        input_config_file=input_valid_file,
        output_config_file=output_config_file,
        output_dir=output_dir,
        scenario_config=ape_config,
        offline=True
    )
    print(os.path.split(output_config_file)[-1])



def run(train_dir_root, valid_dir_root,
        input_train_file, input_valid_file, offline):

    os.makedirs(train_dir_root, exist_ok=True)
    if offline:
        os.makedirs(valid_dir_root, exist_ok=True)

    for ape_config in APEBENCH_STUDIES:
        if offline:
            ape_config.pop("network_config")
            for_validation(
                valid_dir_root,
                input_valid_file,
                ape_config
            )
        else:
            for_training(
                train_dir_root,
                valid_dir_root,
                input_train_file,
                ape_config
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a directory with MelissaxAPEBench configurations."
    )
    parser.add_argument(
        '--train-dir-root', type=str, required=False,
        default=TRAIN_CONFIGS_ROOT
    )
    parser.add_argument(
        '--valid-dir-root', type=str, required=False,
        default=VALID_CONFIGS_ROOT
    )
    parser.add_argument(
        '--input-train-file', type=str, required=False,
        default=TRAIN_FILE
    )
    parser.add_argument(
        '--input-valid-file', type=str, required=False,
        default=VALID_FILE
    )
    parser.add_argument(
        '--offline', action='store_true'
    )

    args = parser.parse_args()
    if args.offline is False:
        assert os.path.exists(args.valid_dir_root), \
            "Training study requires validation_root to be set."

    run(
        args.train_dir_root,
        args.valid_dir_root,
        args.input_train_file,
        args.input_valid_file,
        args.offline
    )
