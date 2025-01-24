import os
import argparse
from modify_json import modify_json

import advection_1d_varying_diff
import broad_comparison_1d


APEBENCH_STUDIES = {
    "diff_adv_1d": advection_1d_varying_diff,
    "broad_compare_1d": broad_comparison_1d
}


INPUT_TRAIN_FILE = "default_config_slurm.json"
INPUT_VALID_FILE = "default_config_offline.json"


def for_training(benchmark_name, ape_config):
    for sampler_type, sampler_suffix in [
        ("uniform", "uniform"),
        ("breed", "breed"),
        ("breed", "breed_mix")
    ]:
        output_config_file, output_dir, validation_dir = \
            APEBENCH_STUDIES[benchmark_name].for_training(
                ape_config, sampler_suffix
            )
        modify_json(
            sampler=sampler_type,
            input_config_file=INPUT_TRAIN_FILE,
            output_config_file=output_config_file,
            output_dir=output_dir,
            scenario_config=ape_config,
            validation_directory=validation_dir,
            use_true_mixing=sampler_suffix == "breed_mix"
        )
        print(os.path.split(output_config_file)[-1])


def for_validation(benchmark_name, ape_config):
    output_config_file, output_dir = \
        APEBENCH_STUDIES[benchmark_name].for_validation(ape_config)

    modify_json(
        input_config_file=INPUT_VALID_FILE,
        output_config_file=output_config_file,
        output_dir=output_dir,
        scenario_config=ape_config,
        offline=True
    )
    print(os.path.split(output_config_file)[-1])


def run(benchmark_name, offline):

    for ape_config in APEBENCH_STUDIES[benchmark_name].CONFIGS:
        if offline:
            ape_config.pop("network_config")
            for_validation(benchmark_name, ape_config)
        else:
            for_training(benchmark_name, ape_config)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Creates a directory with MelissaxAPEBench configurations."
    )
    parser.add_argument(
        '--benchmark-name', type=str, required=True,
        help='specify the name of the benchmark.',
        choices=list(APEBENCH_STUDIES.keys())
    )
    parser.add_argument(
        '--offline', action='store_true'
    )

    args = parser.parse_args()
    run(
        args.benchmark_name,
        args.offline
    )
