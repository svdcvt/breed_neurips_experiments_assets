import os
import argparse
from make_config import create_from, save_configs


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
    parser.add_argument(
        "--study-directory",
        type=str,
        default="./experiments/",
        help="Path to the common studies directory",
    )
    parser.add_argument(
        "--valid-directory",
        type=str,
        default="./datasets/",
        help="Path to the common validation datasets directory",
    )
    parser.add_argument(
        "--subdir",
        type=str,
        default="set",
        help="Subdirectory for created config in the study directory",
    )

    args = parser.parse_args()

    is_test = args.test
    if is_test:
        print("Running in test mode")
    else:
        print("Running in normal mode")

    COMMON_STUDY_DIRECTORY = os.path.join(args.study_directory, args.subdir)
    COMMON_VALID_DIRECTORY = args.study_directory
    GENERAL_SEED = 1234
    
    ##### first we define default common parameters
    scenario_config = {
        "base_scale": 5, # 160 * scale is number of points in mesh
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
        "model_name": "Res", # Res for resnet UNet for Unet
        "num_channels": 12, # depends on the gpu usage, can be more
        "num_blocks": 4, # depends on the gpu usage, can be more
        "lr_start": 5e-3,
        "batch_size": 32, # depends on the gpu usage
        "nb_time_steps": 0.75, # what percentage of a trajectory to use for training compared to validation
        "temporal_horizon": 100, # number of time steps in validation trajectory
    }
    dl_kwargs = { 
        "valid_num_samples": 5, # can be higher if validation set is small, can be lower if it takes too long
        "valid_batch_size": 32 * 10, # factor depends on gpu usage
        "activation": "relu",
        "lr_peak": 5e-4, # peak learning rate (can be lower than start)
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
        "memory_bytes_study": 40 * 1024 * 1024 * 1024,  # Memory for study (buffer) in bytes
        "memory_validation_bytes_file": 40 * 1024 * 1024 * 1024,  # Memory in bytes for the validation file
    }
    active_sampling_kwargs = {
        "regime": "uniform" # one of the regimes: uniform, broad, precise, mixed, soft, no_resampling
        }
    
    # here you can iterate over differnt changing parts
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
            COMMON_VALID_DIRECTORY,
            scenario_kwargs,
            dl_kwargs,
            melissa_kwargs,
            active_sampling_kwargs,
            seed=GENERAL_SEED
        )

        save_configs(
            config_offline,
            config_online,
            cfg_off_path,
            cfg_on_path,
            main_dir,
            test=is_test,
        )
