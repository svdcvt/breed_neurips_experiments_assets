import json
import numpy as np
from dataclasses import dataclass
from typing import Union
import os
import math
import utility as utl
import pprint


DIM = 1
GENERAL_SEED = 1234


@dataclass
class ScenarioConfig:
    def __init__(self, base_scale: int = 5, num_waves: int = 3, **kwargs):
        """
        Scenario configuration for the study.
        Args:
            base_scale (int): Base scale for the scenario.
            num_waves (int): Number of waves in the scenario.
            kwargs: Additional keyword arguments for scenario configuration.
                - mode (str): Mode of the scenario (default: "diff").
                - pde (str): PDE type (default: "ks_cons").
                - diffusion_gamma (float): Diffusion gamma value (default: -2).
                - dispersion_gamma (float): Dispersion gamma value (default: -14).
                - hyp_diffusion_gamma (float): Hyper diffusion gamma value (default: -18).
                - convection_delta (float): Convection delta value (default: -1).
                - convection_sc_delta (float): Convection SC delta value (default: -2).
                - gradient_norm_delta (float): Gradient norm delta value (default: -6).
                - ic_max_one (bool): Whether the maximum is determined by the number of waves (default: False).
        """
        self.base_scale = base_scale
        self.num_points = 160 * self.base_scale
        self.num_spatial_dims = DIM

        self.num_waves = num_waves

        self.l_bounds = [-1.0, 0.0] * self.num_waves
        self.u_bounds = [1.0, round(2 * math.pi, 4)] * self.num_waves

        scenario_mode = kwargs.get("mode", "diff")
        scenario_pde = kwargs.get("pde", "ks_cons")
        if scenario_mode == "diff":
            if scenario_pde == "ks_cons":
                # default is -2 -18 -1
                # easier is -3 -50 -1
                # harder is -2 -16 -1
                diffusion_gamma = kwargs.get("diffusion_gamma", -2)
                hyp_diffusion_gamma = kwargs.get("hyp_diffusion_gamma", -18)
                convection_delta = kwargs.get("convection_delta", -1)
                coefs_ = {
                    "diffusion_gamma": diffusion_gamma * self.base_scale**2,
                    "hyp_diffusion_gamma": hyp_diffusion_gamma * self.base_scale**4,
                    "convection_delta": convection_delta * self.base_scale,
                }
                rel = hyp_diffusion_gamma / diffusion_gamma
                default_rel = rel / 9
            elif scenario_pde == "ks":
                # default is -1.2 -15 -6 , rel 12.5
                # harder is -3.3 -30 -6, rel 9.09
                diffusion_gamma = kwargs.get("diffusion_gamma", -1.2)
                hyp_diffusion_gamma = kwargs.get("hyp_diffusion_gamma", -15)
                gradient_norm_delta = kwargs.get("gradient_norm_delta", -6)
                coefs_ = {
                    "diffusion_gamma": diffusion_gamma * self.base_scale**2,
                    "hyp_diffusion_gamma": hyp_diffusion_gamma * self.base_scale**4,
                    "gradient_norm_delta": gradient_norm_delta * self.base_scale**2,
                }
                rel = hyp_diffusion_gamma / diffusion_gamma
                default_rel = rel / 12.5
            elif scenario_pde == "kdv":
                # default is -14 -9 -2 , rel 0.64
                # harder is -20 -5 -2 , rel 0.25
                dispersion_gamma = kwargs.get("dispersion_gamma", -14)
                hyp_diffusion_gamma = kwargs.get("hyp_diffusion_gamma", -9)
                convection_sc_delta = kwargs.get("convection_sc_delta", -2)
                coefs_ = {
                    "dispersion_gamma": dispersion_gamma * self.base_scale**3,
                    "hyp_diffusion_gamma": hyp_diffusion_gamma * self.base_scale**4,
                    "convection_sc_delta": convection_sc_delta * self.base_scale,
                }
                rel = hyp_diffusion_gamma / dispersion_gamma
                default_rel = rel / 0.64
            else:
                raise NotImplementedError(
                    f"Scenario PDE {scenario_pde} is not implemented."
                )
        else:
            raise NotImplementedError(
                f"Scenario mode {scenario_mode} is not implemented."
            )

        self.scenario_name = f"{scenario_mode}_{scenario_pde}"

        ic_params = ";".join([f"<amp{i+1}>;<phs{i+1}>" for i in range(self.num_waves)])
        # the maximum is determined by number of waves!
        # we assume that the amplitude interval is [-1, 1]
        # so the maximum possible value in IC is 1 * num_waves
        # the generated IC will be divided by that value
        ic_max_one = "true" if kwargs.get("ic_max_one", True) else "false"

        self.ic_config = f"sine_sup;{ic_params};false;{ic_max_one}"

        # scenario identifier
        self.shorthand = f"{self.scenario_name}"
        s = str(round(default_rel, 1)).replace(".", "")
        if default_rel > 1:
            diff_suffix = f"x{s}_easier"
        elif default_rel < 1:
            diff_suffix = f"x{s}_harder"
        else:
            diff_suffix = "default"
        # dynamicity
        self.shorthand += f"__{self.num_waves}w_{diff_suffix}" + (
            "_max1" if ic_max_one == "true" else ""
        )
        # size
        self.shorthand += f"_{self.num_spatial_dims}d_x{self.base_scale}"

        # example:
        # diff_ks_cons__3w_x09_easier_1d_x5
        self.scenario_config = {
            "scenario_name": self.scenario_name,
            "ic_config": self.ic_config,
            "num_points": self.num_points,
        } | coefs_


@dataclass
class DLConfig:
    def __init__(
        self,
        model_name: str = "UNet",
        num_channels: int = 6,
        num_blocks: int = 5,
        lr_start: float = 1e-3,
        batch_size: int = 16,
        nb_time_steps: Union[int, float] = 0.5,
        **kwargs,
    ):
        """
        Deep learning configuration for the study.
        Args:
            model_name (str): Name of the model (default: "UNet").
            num_channels (int): Number of channels in the model (default: 6).
            num_blocks (int): Number of blocks in the model (default: 5).
            lr_start (float): Initial learning rate (default: 1e-3).
            batch_size (int): Batch size for training (default: 16).
            nb_time_steps (int or float): Number of time steps compared to validation (default: 0.5 of validation length = 50 steps).
            kwargs: Additional keyword arguments for deep learning configuration.
                - valid_num_samples (int): Number of samples for validation during training (default: 100).
                - valid_batch_size (int): Batch size for validation (default: twice the batch_size).
                - activation (str): Activation function (default: "relu").
                - lr_peak (float): Peak learning rate (default: lr_start).
                - lr_interval (float): Learning rate interval percentage (default: 0.99).
                - nb_batches_update (int): Number of batches for update (default: 100).
                - temporal_horizon (int): Temporal horizon for the model (default: 100).
        """

        activation = kwargs.get("activation", "relu")
        lr_peak = kwargs.get("lr_peak", lr_start)
        lr_interval = kwargs.get("lr_interval", 0.99)

        self.network_config = f"{model_name};{num_channels};{num_blocks};{activation}"
        self.optim_config = f"adam;warmup_cosine;{lr_start};{lr_peak};{lr_interval}"

        self.nb_batches_update = kwargs.get("nb_batches_update", 100)
        self.valid_batch_size = kwargs.get("valid_batch_size", 2 * batch_size)
        # it should include the initial 0-step
        self.valid_nb_time_steps = kwargs.get("temporal_horizon", 100) + 1
        self.valid_num_samples = kwargs.get("valid_num_samples", 100)
        self.valid_rollout = self.valid_nb_time_steps - 1

        self.batch_size = batch_size
        if isinstance(nb_time_steps, int):
            self.nb_time_steps = nb_time_steps
            self.vision = round((self.valid_nb_time_steps - 1) / nb_time_steps, 2)
        elif isinstance(nb_time_steps, float):
            self.nb_time_steps = round((self.valid_nb_time_steps - 1) * nb_time_steps)
            self.vision = round(nb_time_steps, 2)
        else:
            raise NotImplementedError(
                f"nb_time_steps should be int or float, but got {type(nb_time_steps)}"
            )

        # arch
        self.shorthand = f"{self.network_config.replace(';','_')}"
        # optimization
        numsteps_hc = 10000 if self.valid_nb_time_steps < 150 else 20000
        if isinstance(lr_interval, float):
            lr_pct = lr_interval
            lr_interval = int(lr_interval * (numsteps_hc // 2))
        else:
            lr_interval = int(lr_interval)
            lr_pct = round(lr_interval / (numsteps_hc // 2), 2)
        if lr_start == lr_peak:
            if lr_pct > 0.9:
                self.shorthand += f"__constlr{lr_start:.0e}"
            else:
                self.shorthand += f"__cosinelr{lr_start:.0e}"
        elif lr_start < lr_peak:
            self.shorthand += (
                f"__warmupcosine{lr_start:.1e}_{lr_peak:.1e}_{lr_interval:d}"
            )
        else:
            self.shorthand += f"__decaylr{lr_start:.0e}_{lr_peak:.1e}_{lr_interval:d}"
        self.shorthand += f"__B{self.batch_size}"
        # how far NN sees into the future
        self.shorthand += f"__T{round(self.vision * 100)}p"

        self.shorthand = self.shorthand.replace("-0", "-").replace(".0", "")

        # example:
        # UNet_6_5_relu__warmupcosine1e-03_1e-02_99_B16__T50p


@dataclass
class MelissaConfig:
    def __init__(
        self,
        scenario: ScenarioConfig,
        dl: DLConfig,
        total_nb_simulations_training: int = -1,
        total_nb_simulations_validation: int = -1,
        watermark_num_sim: int = -1,
        buffer_num_sim: int = -1,
        buffer_size_pct: float = 0.1,
        zmq_pct: float = 0.01,
        timeout_minutes: int = 30,
        nb_clients: int = 14,
        timer_delay: int = 5,
        **kwargs,
    ):
        """
        Melissa configuration for the study.
        Args:
            scenario (ScenarioConfig): Scenario configuration object.
            dl (DLConfig): Deep learning configuration object.
            total_nb_simulations_training (int): Total number of simulations for training(default: -1, use available memory_bytes).
            total_nb_simulations_validation (int): Total number of simulations for validation dataset (default: -1, use available memory_bytes).
            watermark_num_sim (int): Number of simulations for watermark (default: nb_clients).
            buffer_num_sim (int): Number of simulations in the buffer (default: use available memory_bytes and buffer_size_pct).
            buffer_size_pct (float): Buffer size percentage from total simulations (default: 0.1).
            zmq_pct (float): ZMQ percentage of buffer (default: 0.01).
            timeout_minutes (int): Timeout of the study in minutes (default: 30).
            nb_clients (int): Number of clients, which affects speed of the study (default: 14).
            timer_delay (int): Timer delay (when to ping clients) in seconds (default: 5).
            kwargs: Additional keyword arguments for Melissa configuration.
                - memory_buffer_bytes_study (int): Memory buffer in bytes for the study (default: (total - bytes(valid_num_samples))*0.8).
                - memory_validation_bytes_file (int): Memory in bytes for the validation file (default: 10Gb).
                - memory_bytes_study (int): Memory in bytes for the study (default: 45Gb).
        """
        # setting buffer watermark
        if watermark_num_sim == -1:
            watermark_num_sim = nb_clients
        self.per_server_watermark = watermark_num_sim * dl.nb_time_steps
        # check total nb simulation, buffer num sim and pct are consistent
        # either we have total nb and pct => auto num_sim, either num_sim and pct => auto total, either pct and memory => auto num_sim and total
        if total_nb_simulations_training != -1 and buffer_num_sim != -1:
            buffer_size_pct = buffer_num_sim / total_nb_simulations_training
            print(
                "The percentage given is ignored, as total number of simulations and buffer size are given."
                f"\nCurrent percentage is {buffer_size_pct*100:.1f}%."
            )
        elif total_nb_simulations_training == -1 and buffer_num_sim != -1:
            total_nb_simulations_training = round(buffer_num_sim / buffer_size_pct)
            print(
                f"The total number of simulations is set to {total_nb_simulations_training} based on given buffer size and percentage."
            )
        elif total_nb_simulations_training != -1 and buffer_num_sim == -1:
            buffer_num_sim = round(total_nb_simulations_training * buffer_size_pct)
            print(
                f"The buffer size is set to {buffer_num_sim} based on given total number of simulations and percentage."
            )
        else:
            print(
                "The total number of simulations and buffer size are not given, using memory budget to set them."
            )

        # checking memory budget and defining buffer size and validation file size
        memory_bytes_study = kwargs.get(
            "memory_bytes_study", min(utl.tob(45), utl.get_available_memory())
        )
        print(f"Memory available for study is {utl.bto(memory_bytes_study):.1f}Gb.")
        memory_validation_bytes_file = kwargs.get(
            "memory_validation_bytes_file", utl.tob(10)
        )
        print(
            f"Memory available for validation file is {utl.bto(memory_validation_bytes_file):.1f}Gb."
        )

        size_sample_val = utl.calculate_data_size(
            scenario.num_points, dl.valid_nb_time_steps - 1, 1
        )
        size_sample_train = utl.calculate_data_size(
            scenario.num_points, dl.nb_time_steps, 1
        )
        print(
            f"Size of one sample is {utl.bto(size_sample_val, 2):.1f}Mb for validation and {utl.bto(size_sample_train, 2):.1f}Mb for training."
        )

        # based on validation sample size and memory budget
        max_total_nb_simulations_validation = round(
            memory_validation_bytes_file / size_sample_val
        )
        print(
            f"Maximum number of samples for validation file is {max_total_nb_simulations_validation} samples."
        )
        if total_nb_simulations_validation == -1:
            # use all memory
            total_nb_simulations_validation = max_total_nb_simulations_validation
        elif total_nb_simulations_validation > max_total_nb_simulations_validation:
            # given nb is too big
            print(
                "WARNING: "
                f"Validation size is capped to {max_total_nb_simulations_validation} samples based on available memory."
                f"\nMemory available for validation file is {utl.bto(memory_validation_bytes_file):.1f}Gb."
                f"\nFor {total_nb_simulations_validation} samples, the size is {utl.bto(size_sample_val * total_nb_simulations_validation):.1f}Gb."
                "\nChange `memory_validation_bytes_file` to increase number of simulations."
            )
            total_nb_simulations_validation = max_total_nb_simulations_validation
        else:
            print(
                f"Validation size is {total_nb_simulations_validation} samples, as given."
            )
        # during training we use only subset of the file
        size_val_set_during_training = size_sample_val * dl.valid_num_samples
        # need to check the number is of samples is not too big for study
        if size_val_set_during_training > memory_bytes_study * 0.2:
            # given nb is too big
            dl.valid_num_samples = round(memory_bytes_study * 0.2 / size_sample_val)
            print(
                "WARNING: "
                f"Validation size is capped to {dl.valid_num_samples} samples based on available memory."
                f"\nMemory available for validation set is {utl.bto(memory_bytes_study * 0.2):.1f}Gb."
                f"\nFor {dl.valid_num_samples} samples, the size is {utl.bto(size_sample_val * dl.valid_num_samples):.1f}Gb."
            )
            size_val_set_during_training = size_sample_val * dl.valid_num_samples

        # check buffer size and memory
        max_memory_buffer = 0.8 * (memory_bytes_study - size_val_set_during_training)
        memory_buffer_bytes_study = kwargs.get(
            "memory_buffer_bytes_study", max_memory_buffer
        )
        if memory_buffer_bytes_study > max_memory_buffer:
            print(
                "WARNING: "
                f"Buffer size is capped to {utl.bto(max_memory_buffer):.1f}Gb based on available memory."
                f"\nMemory available for study is {utl.bto(memory_bytes_study):.1f}Gb and validation set is {utl.bto(size_val_set_during_training):.1f}Gb."
                f"\nAnd buffer is 80% of the remaining memory. But provided value is {utl.bto(memory_buffer_bytes_study):.1f}Gb."
            )
            memory_buffer_bytes_study = max_memory_buffer
        print(
            f"Memory available for buffer is {utl.bto(memory_buffer_bytes_study):.1f}Gb."
        )
        max_buffer_num_sim = round(memory_buffer_bytes_study / size_sample_train)
        if buffer_num_sim == -1:
            buffer_num_sim = max_buffer_num_sim
        elif buffer_num_sim > max_buffer_num_sim:
            print(
                "WARNING: "
                f"Buffer size is capped to {max_buffer_num_sim} samples based on available memory."
                f"\nMemory available for study is {utl.bto(memory_bytes_study):.1f}Gb and validation set is {utl.bto(size_val_set_during_training):.1f}Gb."
                f"\nAnd buffer is 80% of the remaining memory. This is maximum {max_buffer_num_sim} number of simulations of size {utl.bto(size_sample_train, 2):.2f}Mb."
                f"\nBut provided value is bigger: {buffer_num_sim}."
            )
            buffer_num_sim = max_buffer_num_sim

        if total_nb_simulations_training == -1:
            total_nb_simulations_training = round(buffer_num_sim / buffer_size_pct)
        elif total_nb_simulations_training > round(buffer_num_sim / buffer_size_pct):
            print(
                "WARNING: "
                f"Total number of simulations is not consistent with buffer size and percentage of buffer from total."
                f"\nMemory available for buffer is {utl.bto(memory_buffer_bytes_study):.1f}Gb which is {buffer_num_sim} samples."
                f"\nAnd buffer is {buffer_size_pct * 100:.0f}% of the total number of simulations."
                f"\nThis makes total {round(buffer_num_sim / buffer_size_pct)} simulations."
                f"\nBut provided value is bigger: {total_nb_simulations_training}."
                f"\nCapping to {round(buffer_num_sim / buffer_size_pct)} simulations to have consistent percentage."
            )
            total_nb_simulations_training = round(buffer_num_sim / buffer_size_pct)

        self.total_nb_simulations_offline = total_nb_simulations_validation
        self.total_nb_simulations_online = total_nb_simulations_training
        self.buffer_size = buffer_num_sim * dl.nb_time_steps

        self.zmq_hwm = round(zmq_pct * self.buffer_size)
        self.buffer_size -= self.zmq_hwm
        self.valid_num_samples = dl.valid_num_samples

        self.timeout_minutes = timeout_minutes
        self.nb_clients = nb_clients
        # 1 second recommended but to "imitate" slow simulations can be increased
        self.timer_delay = timer_delay

        # capacity
        self.shorthand = f"{buffer_num_sim}BUF_{watermark_num_sim}WM"
        # speed
        self.shorthand += f"__{self.timer_delay}TD_{self.nb_clients}CL"

        # example:
        # 20BUF_10WM__5TD_14CL


@dataclass
class ActiveSamplingConfig:
    def __init__(
        self,
        scenario: ScenarioConfig,
        dl: DLConfig,
        melissa: MelissaConfig,
        regime: str = "uniform",
        **kwargs,
    ):
        # instead of iterating over many numerical hyperparameters
        # we establish sets of hyperparameters and give them names

        # nn_updates, min_nb_finished_simulations - reactiveness : low, high
        # high
        # min_nb_finished_simulations: 2 * watermark in the buffer = allow first resampling almost as soon as nn_updates is reached
        # nn_updates: 100 updates over the expected number of batches
        # low
        # min_nb_finished_simulations: buffer size samples = allow first resampling only after the buffer is full
        # nn_updates: 10 updates over the expected number of batches

        # sliding_window_size - memory: short, long
        # could be defined through buffer size,
        # short - not more than buffer samples,
        # long - twice more than buffer samples
        # short: 1 * BS, long: 2 * BS

        # delta_loss_min_nb_time_steps - impulsiveness: low, high
        # low: 0.9 of trajectory, high: 0.25 of trajectory

        # 1. memory short + impulsiveness low + reactiveness low
        # 2. memory long + impulsiveness high + reactiveness high

        #  -----------------

        # sigma -> concentration: narrow, wide
        # narrow: 0.05, wide: 0.1
        # this is the ratio between "parent neighbourhood" and parameter interval length
        # where "parent neighbourhood" = 3 sigma = 86.6% of the normal distribution

        # (start, end, breakpoint) - epsilon-greedy schedule: explorative, exploitative
        # explorative: 0.5, 0.75, 5
        # exploitative: 0.75, 0.9, 3

        # 1. narrow + explorative
        # 2. wide + exploitative

        #  -----------------

        # Two regimes (+ 2 baselines):
        # 1. Precise: narrow + explorative + memory short + impulsiveness low + reactiveness low
        # 2. Broad: wide + exploitative + memory long + impulsiveness high + reactiveness high
        # 3. Uniform: no proposal, but we use the same parameters as in the precise regime
        # 4. No resampling: nn_updates = -1

        nb_expected_batches = round(
            melissa.total_nb_simulations_online * dl.nb_time_steps / dl.batch_size * 1.5
        )
        pars_interval_length = [
            scenario.u_bounds[i] - scenario.l_bounds[i]
            for i in range(scenario.num_waves * 2)
        ]
        buffer_samples = round(melissa.buffer_size // dl.nb_time_steps)

        if regime == "precise":
            sigma = 0.05
            start = 0.5
            end = 0.75
            breakpoint_ = 5
            sliding_window_size = buffer_samples
            fitness_min_nb_time_steps = round(0.9 * dl.nb_time_steps)
            min_nb_finished_simulations = buffer_samples
            resample_each_nn_updates = buffer_samples * 5
        elif regime == "broad":
            sigma = 0.1
            start = 0.75
            end = 0.9
            breakpoint_ = 3
            sliding_window_size = 2 * buffer_samples
            fitness_min_nb_time_steps = round(0.25 * dl.nb_time_steps)
            min_nb_finished_simulations = round(
                (melissa.per_server_watermark // dl.nb_time_steps) * 2
            )
            resample_each_nn_updates = buffer_samples * 2
        elif regime == "mixed":
            sigma = 0.1
            start = 0.5
            end = 0.9
            breakpoint_ = 3
            sliding_window_size = buffer_samples
            fitness_min_nb_time_steps = round(0.33 * dl.nb_time_steps)
            min_nb_finished_simulations = round(
                (melissa.per_server_watermark // dl.nb_time_steps) * 2
            )
            resample_each_nn_updates = buffer_samples * 3
        elif regime == "soft":
            sigma = 0.1
            start = 0.75
            end = 0.5
            breakpoint_ = 5
            sliding_window_size = buffer_samples * 3
            fitness_min_nb_time_steps = round(0.5 * dl.nb_time_steps)
            min_nb_finished_simulations = round(0.9 * buffer_samples)
            resample_each_nn_updates = buffer_samples * 1
        elif regime == "uniform":
            # same as mixed but r-value = 0
            sigma = 0.1
            start = 0.0
            end = 0.0
            breakpoint_ = 3
            sliding_window_size = buffer_samples
            fitness_min_nb_time_steps = round(0.33 * dl.nb_time_steps)
            min_nb_finished_simulations = round(
                (melissa.per_server_watermark // dl.nb_time_steps) * 2
            )
            resample_each_nn_updates = buffer_samples * 3
        elif regime == "no_resampling":
            sigma = 0.5
            start = 0.0
            end = 0.0
            breakpoint_ = 1
            sliding_window_size = buffer_samples
            fitness_min_nb_time_steps = round(0.9 * dl.nb_time_steps)
            min_nb_finished_simulations = buffer_samples
            resample_each_nn_updates = -1
        else:
            if regime == "custom" and kwargs is None:
                raise ValueError("Custom regime requires kwargs to be set")

        self.config_dict = dict()
        #  each X batches resampling is triggered
        self.config_dict["nn_updates"] = kwargs.get(
            "nn_updates", resample_each_nn_updates
        )
        #  the threshold for the first resampling
        self.config_dict["min_nb_finished_simulations"] = kwargs.get(
            "min_nb_finished_simulations", min_nb_finished_simulations
        )
        #  the threshold for bein a parent
        self.config_dict["delta_loss_min_nb_time_steps"] = kwargs.get(
            "delta_loss_min_nb_time_steps", fitness_min_nb_time_steps
        )
        #  when to stop resampling
        self.config_dict["non_resampling_threshold"] = 1

        self.config_dict["breed_params"] = {
            "non_breed_sampling_strategy": "random",
            "sigma": [
                round(kwargs.get("sigma", sigma) * length, 4)
                for length in pars_interval_length
            ],
            "start": kwargs.get("start", start),
            "end": kwargs.get("end", end),
            "breakpoint": kwargs.get("breakpoint", breakpoint_),
            "sliding_window_size": kwargs.get(
                "sliding_window_size", sliding_window_size
            ),
            "use_true_mixing": True,
            "log_extra": False,
        }

        if len(kwargs) > 1:
            self.shorthand = regime + "_custom"
        else:
            self.shorthand = regime


class StudyConfig:
    def __init__(
        self,
        scenario: ScenarioConfig,
        dl: DLConfig,
        melissa: MelissaConfig,
        active_sampling: ActiveSamplingConfig,
        common_study_directory: str,
        common_valid_directory: str,
        default_configs_file="default_configs.json",
        seed: int = GENERAL_SEED,
    ):
        self.scenario = scenario
        self.dl = dl

        self.melissa = melissa
        self.active_sampling = active_sampling
        self.seed = seed

        with open(default_configs_file, "r") as f:
            self.default_configs = json.load(f)

        self.validation_subname = f"{self.scenario.shorthand}"
        self.validation_directory = os.path.join(
            common_valid_directory, self.validation_subname
        )
        if os.path.exists(
            os.path.join(
                self.validation_directory, "trajectories", "all_trajectories.npy"
            )
        ):
            print(
                f"WARNING: Validation file already exists in: {self.validation_directory}. Most probably, you don't need to run the offline study again."
            )
            self.validation_exists_flag = True
        else:
            self.validation_exists_flag = False

        self.study_components = [
            self.scenario.shorthand.split("__"),
            self.melissa.shorthand.split("__"),
            self.dl.shorthand.split("__"),
            self.active_sampling.shorthand,
        ]

        self.study_name = "__".join(
            [
                self.scenario.shorthand,
                self.melissa.shorthand,
                self.dl.shorthand,
                self.active_sampling.shorthand,
            ] + ([] if seed == GENERAL_SEED else [f"seed{seed}"])
        )
        subdir = "/".join(
            [
                self.scenario.shorthand,
                self.melissa.shorthand,
                self.dl.shorthand,
                self.active_sampling.shorthand,
            ] + ([] if seed == GENERAL_SEED else [f"seed{seed}"])
        )

        study_directory = os.path.join(common_study_directory, subdir)
        self.study_directory = utl.get_next_dir(study_directory)
        print(f"Study directory: {self.study_directory}")

        if "slurm" in self.default_configs["offline"]["launcher_config"]["scheduler"]:
            self.sched_t = "slurm"
        else:
            self.sched_t = "mpi"

    def generate_offline(self):
        if self.validation_exists_flag:
            print(
                f"WARNING: Validation file already exists in: {self.validation_directory}. Check the files."
            )

        config = utl.deep_update(
            self.default_configs["offline"],
            {
                "output_dir": self.validation_directory,
                "study_options": {
                    "scenario_config": self.scenario.scenario_config
                    | {"network_config": "MLP;1;1;relu"},
                    "parameter_sweep_size": self.melissa.total_nb_simulations_offline,
                    "nb_time_steps": self.dl.valid_nb_time_steps - 1,  # not counting IC
                    "nb_parameters": self.scenario.num_waves * 2,
                    "l_bounds": self.scenario.l_bounds,
                    "u_bounds": self.scenario.u_bounds,
                    "seed": GENERAL_SEED * 100 + 42,
                },
                "launcher_config": {
                    # "job_limit": self.melissa.nb_clients + 1,
                    "http_port": np.random.randint(8000, 9000),
                },
            },
        )

        config_path = f"config_offline_{self.validation_subname}{'_copy' if self.validation_exists_flag else ''}_{self.sched_t}.json"
        config["client_config"]["preprocessing_commands"].append(
            f"export CONFIG_FILE={config_path}"
        )
        return config, config_path

    def generate_online(self):
        config = utl.deep_update(
            self.default_configs["online"],
            {
                "output_dir": self.study_directory,
                "study_options": {
                    "scenario_config": self.scenario.scenario_config
                    | {
                        "network_config": self.dl.network_config,
                        "optim_config": self.dl.optim_config,
                    },
                    "parameter_sweep_size": self.melissa.total_nb_simulations_online,
                    "nb_time_steps": self.dl.nb_time_steps,
                    "nb_parameters": self.scenario.num_waves * 2,
                    "l_bounds": self.scenario.l_bounds,
                    "u_bounds": self.scenario.u_bounds,
                    "seed": self.seed,
                    "zmq_hwm": self.melissa.zmq_hwm,
                },
                "active_sampling_config": self.active_sampling.config_dict,
                "dl_config": {
                    "validation_directory": os.path.join(
                        self.validation_directory, "trajectories"
                    ),
                    "validation_file": "all_trajectories.npy",
                    "valid_rollout": self.dl.valid_rollout,
                    "valid_batch_size": self.dl.valid_batch_size,
                    "valid_nb_time_steps": self.dl.valid_nb_time_steps,
                    "valid_num_samples": self.melissa.valid_num_samples,
                    "nb_batches_update": self.dl.nb_batches_update,
                    "batch_size": self.dl.batch_size,
                    "per_server_watermark": self.melissa.per_server_watermark,
                    "buffer_size": self.melissa.buffer_size,
                },
                "launcher_config": {
                    "job_limit": self.melissa.nb_clients + 2,
                    "timer_delay": self.melissa.timer_delay,
                    "http_port": np.random.randint(8000, 9000),
                },
            },
        )
        import datetime
        seconds = self.melissa.timeout_minutes * 60
        if self.sched_t == "slurm":
            timeout = [f"--time={datetime.timedelta(seconds=seconds)}"]
        else:
            timeout = ["--timeout", str(seconds)]
        config["launcher_config"]["scheduler_arg_server"] = config["launcher_config"][
            "scheduler_arg_server"
        ] + timeout
        config_path = f"config_{self.study_name}_{self.sched_t}.json"
        config["client_config"]["preprocessing_commands"].append(
            f"export CONFIG_FILE={config_path}"
        )
        return config, config_path


def test_scenario_config(scenario_config):
    scenario = ScenarioConfig(**scenario_config)
    print(f"Created scenario: {scenario.shorthand}")
    return scenario


def test_dl_config(dl_config):
    dl = DLConfig(**dl_config)
    print(f"Created DL config: {dl.shorthand}")
    return dl


def test_melissa_config(melissa_config, scenario, dl):
    melissa = MelissaConfig(scenario, dl, **melissa_config)
    print(f"Created Melissa config: {melissa.shorthand}")
    return melissa


def test_active_sampling_config(active_sampling_config, scenario, dl, melissa):
    active = ActiveSamplingConfig(
        **active_sampling_config, scenario=scenario, dl=dl, melissa=melissa
    )
    print(f"Created Active Sampling config: {active.shorthand}")
    return active


def test_study_config(scenario, dl, melissa, active):
    study = StudyConfig(
        scenario,
        dl,
        melissa,
        active,
        common_study_directory="./",
        common_valid_directory="./",
    )
    print(f"Created Study config: {study.study_directory}")
    return study


def test_config_generation():
    """Test configuration generation with fixed values"""
    scenario_config = {
        "base_scale": 5,
        "num_waves": 3,
        "mode": "diff",
        "pde": "ks_cons",
        "diffusion_gamma": -3,
        "hyp_diffusion_gamma": -50,
        "convection_delta": -1,
    }
    dl_config = {
        "model_name": "UNet",
        "num_channels": 6,
        "num_blocks": 5,
        "lr_start": 1e-3,
        "batch_size": 16,
        "valid_batch_size": 32,
        "valid_num_samples": 100,
        "nb_time_steps": 0.50,
    }
    melissa_config = {
        "total_nb_simulations": 3000,
        "per_server_watermark_num_sim": 10,
        "buffer_size_pct": 0.05,
        "zmq_pct": 0.01,
        "timeout_minutes": 60,
        "nb_clients": 14,
        "timer_delay": 5,
    }
    active_sampling_config = {
        "regime": "precise",
        "nn_updates": 100,
    }
    try:
        scenario = test_scenario_config(scenario_config)
        dl = test_dl_config(dl_config)
        melissa = test_melissa_config(melissa_config, scenario, dl)
        active = test_active_sampling_config(
            active_sampling_config, scenario, dl, melissa
        )
        study = test_study_config(scenario, dl, melissa, active)
        config_offline, cfg_off_path = study.generate_offline()
        config_online, cfg_on_path = study.generate_online()

        with open(cfg_off_path, "w") as f:
            json.dump(config_offline, f, indent=4)
        with open(cfg_on_path, "w") as f:
            json.dump(config_online, f, indent=4)
        return cfg_off_path, cfg_on_path

    except Exception as e:
        print(f"Failed to create config: {str(e)}")
        raise


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
    conf_off, conf_on = test_config_generation()
