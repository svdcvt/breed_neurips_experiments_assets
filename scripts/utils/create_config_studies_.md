# Experiments configurations creation

Part of the `create_configs_set.py` for the study over all PDEs:

```python
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
        "lr_peak": 1e-4,
        "lr_interval": 5000,
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
            melissa_config["buffer_num_sim"] = 100
        else:
            melissa_config["timeout_minutes"] = 60

        for regime in ["uniform", "mixed", "precise", "broad", "no_resampling", "soft"]:
            ....
```

Part of the `create_configs_set.py` for the study of random seeds:

```python
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
        "lr_peak": 1e-4,
        "lr_interval": 5000,
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

    set_pde = pd.read_csv("./pde_set_few.csv")

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
            melissa_config["buffer_num_sim"] = 100
        else:
            melissa_config["timeout_minutes"] = 30


        for regime in ["mixed", "uniform", "broad", "no_resampling"]:
            for seed in [777, 2025, 424242, 1111, 0]:
                ....
```

