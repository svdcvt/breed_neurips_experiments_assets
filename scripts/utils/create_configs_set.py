from make_config import ScenarioConfig, DLConfig, MelissaConfig, ActiveSamplingConfig, StudyConfig
import json
import os
import pprint
import sys
import utility as utl


COMMON_STUDY_DIRECTORY = "/home/dymchens-ext/apebench_test/experiments/set/"

def create_from(
    scenario_config, dl_config, melissa_config,
    scenario_kwargs=dict(), dl_kwargs=dict(), melissa_kwargs=dict(),
    active_sampling_kwargs=dict()
    ):
    scenario = ScenarioConfig(**scenario_config, **scenario_kwargs)
    dl = DLConfig(**dl_config, **dl_kwargs)
    melissac = MelissaConfig(scenario, dl, **melissa_config, **melissa_kwargs)
    if len(active_sampling_kwargs) == 0:
        active_sampling_kwargs["regime"] = "uniform"
    active_sampling = ActiveSamplingConfig(scenario, dl, melissac, **active_sampling_kwargs)
    study_config = StudyConfig(scenario, dl, melissac, active_sampling, common_study_directory=COMMON_STUDY_DIRECTORY)
    config_online, cfg_on_path = study_config.generate_online()
    if not study_config.validation_exists_flag:
        config_offline, cfg_off_path = study_config.generate_offline()
        return config_offline, cfg_off_path, config_online, cfg_on_path, study_config.study_directory
    return None, None, config_online, cfg_on_path, os.path.dirname(study_config.study_directory)

def save_configs(
    config_offline, config_online, cfg_off_path, cfg_on_path, main_dir, test=False
    ):
    """
    Save the generated configurations to the specified directory.
    """
    if not test:
        os.makedirs(main_dir, exist_ok=True)
        if config_offline is not None:
            with open(os.path.join(main_dir, cfg_off_path), 'w') as f:
                json.dump(config_offline, f, indent=4)
        with open(os.path.join(main_dir, cfg_on_path), 'w') as f:
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
    # the script should be run from hostname bigfoot, check that
    if os.uname()[1] != "bigfoot":
        raise RuntimeError("This script should be run from hostname bigfoot")

    scenario_config = {
                "base_scale": 5,
                "num_waves": 3
    }
    scenario_kwargs = {
            "mode": "diff",
            "pde": "ks_cons",
            "diffusion_gamma": -3,
            "hyp_diffusion_gamma": -50,
            "convection_delta": -1
    }

    dl_config = {
            "model_name": "UNet",
            "num_channels": 6,
            "num_blocks": 5,
            "lr_start": 1e-3,
            "batch_size": 256,
            "nb_time_steps": 0.75
    }

    dl_kwargs = {
            "valid_num_samples": 100,
            "valid_batch_size": 256 * 2,
            "activation": "relu",
            "lr_peak": 1e-3,
            "lr_interval": 2500,
            "nb_batches_update": 200
    }

    melissa_config = {
            "total_nb_simulations_training": 2000,
            "total_nb_simulations_validation": 1500,
            "watermark_num_sim": 10,
            "buffer_num_sim": 100,
            "buffer_size_pct": 0.05,
            "zmq_pct": 0.01,
            "timeout_minutes": 50,
            "nb_clients": 6,
            "timer_delay": 2
    }
    melissa_kwargs = {
        # "memory_buffer_bytes_study":  ,
        # "memory_validation_bytes_file": utl.tob(1),
        # "memory_bytes_study": utl.tob(30)
        }

    active_sampling_kwargs = {
            # "regime": "precise",
            # "nn_updates": 100,
            # "min_nb_finished_simulations": 100,
            # "delta_loss_min_nb_time_steps": 90,
            # "sigma": 0.05,
            # "start": 0.9,
            # "end": 0.9,
            # "breakpoint": 3,
            # "sliding_window_size": 50,
    }
    if len(sys.argv) > 1:
        is_test = sys.argv[1] == "test"
        if is_test:
            print("Running in test mode")
    else:
        is_test = False
        print("Running in normal mode")

    for regime in ["uniform", "precise", "broad", "no_resampling"]:
        active_sampling_kwargs["regime"] = regime
        config_offline, cfg_off_path, config_online, cfg_on_path, main_dir = create_from(
            scenario_config, dl_config, melissa_config,
            scenario_kwargs, dl_kwargs, melissa_kwargs, active_sampling_kwargs
        )
        save_configs(config_offline, config_online, cfg_off_path, cfg_on_path, main_dir, test=is_test)
