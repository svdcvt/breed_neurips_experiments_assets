import os
import pandas as pd
import argparse

from make_config import create_from, save_configs

GENERAL_SEED = 1234

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

    subdir="set"

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

    scenario_config = {"base_scale": 5, "num_waves": 3}
    scenario_kwargs = {
        "mode": "diff",
        "pde": "ks_cons",
        "diffusion_gamma": -3,
        "hyp_diffusion_gamma": -50,
        "convection_delta": -1,
        "ic_max_one": True,
    }
    dl_config = {
        "model_name": "UNet",
        "num_channels": 6,
        "num_blocks": 5,
        "lr_start": 1e-3,
        "batch_size": 256,
        "nb_time_steps": 0.75,
        "temporal_horizon": 100,
    }
    dl_kwargs = {
        "valid_num_samples": 100,
        "valid_batch_size": 256 * 4,
        "activation": "relu",
        "lr_peak": 1e-3,
        "lr_interval": 2500,
        "nb_batches_update": 200,
    }
    melissa_config = {
        "total_nb_simulations_training": 2000,
        "total_nb_simulations_validation": 1500,
        "watermark_num_sim": 10,
        "buffer_num_sim": 100,
        "buffer_size_pct": 0.05,
        "zmq_pct": 0.01,
        "timeout_minutes": 60,
        "nb_clients": 8,
        "timer_delay": 2,
    }
    melissa_kwargs = {}
    active_sampling_kwargs = {"regime": "uniform"}

    set_pde = pd.read_csv("./pde_set.csv")

    for row in set_pde.iterrows():
        row = row[1]
        scenario_config["num_waves"] = row["num_waves"]
        scenario_kwargs["pde"] = row["pde"]
        scenario_kwargs["diffusion_gamma"] = row["diffusion_gamma"]
        scenario_kwargs["dispersion_gamma"] = row["dispersion_gamma"]
        scenario_kwargs["hyp_diffusion_gamma"] = row["hyp_diffusion_gamma"]
        scenario_kwargs["convection_delta"] = row["convection_delta"]
        scenario_kwargs["convection_sc_delta"] = row["convection_sc_delta"]
        scenario_kwargs["ic_max_one"] = row["ic_max_one"]
        dl_config["temporal_horizon"] = row["temporal_horizon"]
        if row["temporal_horizon"] == 200:
            melissa_config["timeout_minutes"] = 120
            melissa_config["total_nb_simulations_training"] = 4000
            melissa_config["buffer_num_sim"] = 100  # pct even less 0.025
        else:
            melissa_config["timeout_minutes"] = 60

        for regime in ["uniform", "mixed", "precise", "broad", "no_resampling", "soft"]:
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
                    print("be sure you created offline configs on DAHU and going to run them on DAHU cluster!!!")    
                    save_configs(
                        config_offline,
                        config_online,
                        cfg_off_path,
                        cfg_on_path,
                        main_dir,
                        test=is_test,
                    )
