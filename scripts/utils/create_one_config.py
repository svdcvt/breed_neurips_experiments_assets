from make_config import (
    ScenarioConfig,
    DLConfig,
    MelissaConfig,
    ActiveSamplingConfig,
    StudyConfig,
)
import json
import os
import pprint
import argparse

GENERAL_SEED = 1234

def create_from(
    scenario_config,
    dl_config,
    melissa_config,
    default_configs_file,
    common_study_directory,
    common_valid_directory,
    scenario_kwargs=dict(),
    dl_kwargs=dict(),
    melissa_kwargs=dict(),
    active_sampling_kwargs=dict(),
    seed=GENERAL_SEED
):
    scenario = ScenarioConfig(**scenario_config, **scenario_kwargs)
    dl = DLConfig(**dl_config, **dl_kwargs)
    melissac = MelissaConfig(scenario, dl, **melissa_config, **melissa_kwargs)
    if len(active_sampling_kwargs) == 0:
        active_sampling_kwargs["regime"] = "uniform"
    active_sampling = ActiveSamplingConfig(
        scenario, dl, melissac, **active_sampling_kwargs
    )
    study_config = StudyConfig(
        scenario,
        dl,
        melissac,
        active_sampling,
        common_study_directory, # what is going to be in the config, important to have /home/,,, always??
        common_valid_directory,
        default_configs_file,
        seed=seed
    )
    config_online, cfg_on_path = study_config.generate_online()
    if not study_config.validation_exists_flag:
        config_offline, cfg_off_path = study_config.generate_offline()
        return (
            config_offline,
            cfg_off_path,
            config_online,
            cfg_on_path,
            os.path.dirname(study_config.study_directory),
        )
    return (
        None,
        None,
        config_online,
        cfg_on_path,
        os.path.dirname(study_config.study_directory),
    )


def save_configs(
    config_offline, config_online, cfg_off_path, cfg_on_path, main_dir, test=False
):
    """
    Save the generated configurations to the specified directory.
    """
    if not test:
        os.makedirs(main_dir, exist_ok=True)
        if config_offline is not None:
            print(os.path.join(main_dir, cfg_off_path))
            with open(os.path.join(main_dir, cfg_off_path), "w") as f:
                json.dump(config_offline, f, indent=4)
        if config_online is not None:
            print(os.path.join(main_dir, cfg_on_path))
            with open(os.path.join(main_dir, cfg_on_path), "w") as f:
                json.dump(config_online, f, indent=4)
    else:
        print("=" * 120)
        print("Offline config path:", cfg_off_path)
        pprint.pp(config_offline, width=120)
        print("=" * 120)
        print("Online config path:", cfg_on_path)
        pprint.pp(config_online, width=120)
        print("=" * 120)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create configuration sets.")
    parser.add_argument(
        "--test", action="store_true", help="Run in test mode."
    )
    parser.add_argument(
        "--default_configs_file",
        type=str,
        default="default_configs.json",
        help="Path to the default configurations file",
    )
    args = parser.parse_args()

    is_test = args.test
    if is_test:
        print("Running in test mode")
    else:
        print("Running in normal mode")

    # subdir="set"
    subdir = "BIG"

    if os.uname()[1] == "bigfoot":
        COMMON_STUDY_DIRECTORY = "/home/dymchens-ext/apebench_test/experiments/{}/".format(subdir)
        COMMON_VALID_DIRECTORY = (
            "/bettik/PROJECTS/pr-melissa/COMMON/datasets/apebench_val/"
        )
    elif "dahu" in os.uname()[1]:
        COMMON_STUDY_DIRECTORY = (
            "/home/dymchens-ext/apebench_test/experiments/{}/".format(subdir)
        )
        COMMON_VALID_DIRECTORY = (
            "/bettik/PROJECTS/pr-melissa/COMMON/datasets/apebench_val/"
        )
    elif "leonardo" in os.uname()[1]:
        COMMON_STUDY_DIRECTORY = (
            "/leonardo_work/EUHPC_D23_125/abhishek/apebench_test/experiments/{}/".format(subdir)
        )
        COMMON_VALID_DIRECTORY = (
            "/leonardo_work/EUHPC_D23_125/datasets/apebench_val/"
        )

    scenario_config = {
        "base_scale": 5000, # 160 * scale is number of points in mesh
        "num_waves": 3
        }
    scenario_kwargs = {
        "mode": "diff",
        "pde": "ks_cons",
        "diffusion_gamma": -2.0,
        "hyp_diffusion_gamma": -18.0,
        "convection_delta": -1,
        "ic_max_one": True,
    }
    dl_config = {
        "model_name": "Res",
        "num_channels": 12, # depends on the gpu usage, can be more
        "num_blocks": 4, # depends on the gpu usage, can be more
        "lr_start": 5e-3,
        "batch_size": 32, # depends on the gpu usage
        "nb_time_steps": 0.75, # what percentage of a trajectory to use for training compared to validation
        "temporal_horizon": 100, # number of time steps in validation trajectory
    }
    dl_kwargs = { 
        "valid_num_samples": 5, # can be higher if validation is in parallel, can be lower if it takes too long
        "valid_batch_size": 32 * 10, # factor depends on gpu usage
        "activation": "relu",
        "lr_peak": 5e-4, # peak learning rate (can be lowe r than start)
        "lr_interval": 10000, # depends on the network speed, number of iteration for decaying lr
        "nb_batches_update": 200, # how often to do validation
    }
    melissa_config = {
        "total_nb_simulations_training": -1, 
        # total nb simulations for training, if -1 calculated based on buffer size and pct
        "total_nb_simulations_validation": 100,
        # if -1 calculated based on memory given
        "watermark_num_sim": 10, # watermark size in nb trajectories
        "buffer_num_sim": -1, # number of simulations in buffer, if -1 calculated based on memory
        "buffer_size_pct": 0.05, # pct from total number of simulations to decide buffer size
        "zmq_pct": 0.01,
        "timeout_minutes": 60 * 6, # how long the study will run overall
        "nb_clients": 8, # change accordingly
        "timer_delay": 2,
    }
    melissa_kwargs = {
        "memory_bytes_study": 40 * 1024 * 1024 * 1024,  # Memory for buffer in bytes
        "memory_validation_bytes_file": 40 * 1024 * 1024 * 1024,  # Memory in bytes for the validation file
    }
    active_sampling_kwargs = {
        "regime": "uniform"
        }


    for regime in ["broad", "mixed", "soft", "no_resampling"]:
        active_sampling_kwargs["regime"] = regime
        (
            config_offline,
            cfg_off_path,
            config_online,
            cfg_on_path,
            main_dir,
        ) = create_from(

            scenario_config,
            dl_config,
            melissa_config,
            args.default_configs_file,
            COMMON_STUDY_DIRECTORY,
            # what is going to be in the config so depends on where it is going to be ran
            COMMON_VALID_DIRECTORY,
            scenario_kwargs,
            dl_kwargs,
            melissa_kwargs,
            active_sampling_kwargs,
            seed=GENERAL_SEED
        )
        if "dahu" in os.uname()[1]:
            save_configs(
                config_offline,
                config_online,
                cfg_off_path,
                cfg_on_path,
                main_dir.replace("/home/", "/home-bigfoot/"),
                test=is_test,
            )
            save_configs(
                config_offline,
                None,
                cfg_off_path,
                None,
                "/home/dymchens-ext/apebench_test/experiments/{}/".format(subdir),
                test=is_test,
            )
        else:
            save_configs(
                config_offline,
                config_online,
                cfg_off_path,
                cfg_on_path,
                main_dir,
                test=is_test,
            )
