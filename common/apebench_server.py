# flake8: noqa
import os
import time
import logging
import gc

from typing_extensions import override
import numpy as np
import jax.numpy as jnp
import jax
import matplotlib.pyplot as plt

import pdequinox as pdeqx
import equinox as eqx
import exponax as ex

from melissa.server.offline_server import OfflineServer
from melissa.server.deep_learning import active_sampling
from melissa.server.deep_learning.active_sampling.active_sampling_server import (
    ExperimentalDeepMelissaActiveSamplingServer
)
from dl_utils import merge_npy_files

import dl_utils
import plot_utils as putils
import monitoring_utils as mutils

from sampler import get_sampler_class_type
from scenarios import MelissaSpecificScenario

logger = logging.getLogger("melissa")
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)


class CommonInitMixIn:
    def __init__(self, config_dict, is_valid=False):
        super().__init__(config_dict)
        if not is_valid:
            if self.world_rank == self.breed_rank:
                time.sleep(5)
            a = jnp.zeros((1, 1, 1))

        # Reading the config file
        study_options = config_dict["study_options"]
        scenario_config = study_options["scenario_config"]
        l_bounds, u_bounds = study_options["l_bounds"], study_options["u_bounds"]
        for i in range(len(l_bounds)):
            if isinstance(l_bounds[i], str):
                l_bounds[i] = eval(l_bounds[i])
            if isinstance(u_bounds[i], str):
                u_bounds[i] = eval(u_bounds[i])
        
        if not is_valid:
            if "optim_config" not in scenario_config:
                scenario_config["optim_config"] = "adam;warmup_cosine;1e-5;1e-3;0.05"
            optim_config = scenario_config["optim_config"].split(";")
            optim_name = optim_config[0]
            expected_batches_min = int(
                study_options["parameter_sweep_size"] * study_options["nb_time_steps"] / config_dict["dl_config"]["batch_size"]
            )
            # num_training_steps = expected_batches_min * 2 # TODO this is very approximate!
            # TODO FIXME (i dont know how...)
            num_training_steps = 20000

            scheduler_name = optim_config[1]
            if scheduler_name == "warmup_cosine":
                scheduler_start = float(optim_config[2])
                scheduler_peak = float(optim_config[3])
                # scheduler_warmup = int(float(optim_config[4]) * expected_batches_min)
                scheduler_warmup = 10000
                # 'adam;10_000;warmup_cosine;0.0;1e-3;2_000'
                scenario_config["optim_config"] = (
                    f"{optim_name};{num_training_steps};{scheduler_name};"
                    f"{scheduler_start};{scheduler_peak};{scheduler_warmup}"
                )
            elif scheduler_name == "constant":
                scheduler_start = float(optim_config[2])
                scenario_config["optim_config"] = (
                    f"{optim_name};{num_training_steps};{scheduler_name};"
                    f"{scheduler_start}"
                )
            else:
                logger.warning(
                    f"Scheduler {scheduler_name} not recognized. "
                    f"Using default warmup_cosine scheduler."
                )
                scenario_config["optim_config"] = (
                    f"{optim_name};{num_training_steps};warmup_cosine;"
                    f"1e-5;1e-3;{int(0.05 * expected_batches_min)}"
                )

        # Setting configurations
        self.scenario = MelissaSpecificScenario(**scenario_config)
        self.mesh_shape = self.scenario.get_shape()
        # (B, 1, 160) -> (1, 2); (B, 1, 160, 160) -> (1, 2, 3)...
        self.mesh_axes = tuple(range(1, self.scenario.num_spatial_dims + 2)) 
        self.breed_params = config_dict.get('active_sampling_config', dict()).get("breed_params", dict())
        self.sampler_t = get_sampler_class_type(
            ic_type=scenario_config["ic_config"].split(";")[0],
            use_classic=is_valid
        )
        print(self.breed_params)
        self.set_parameter_sampler(
            sampler_t=self.sampler_t,
            ic_config=scenario_config["ic_config"],
            is_valid=is_valid,
            **self.breed_params,
            seed=study_options["seed"],
            l_bounds=l_bounds,
            u_bounds=u_bounds,
            dtype=np.float32
        )

        # self.experimental_monitoring = False


class APEBenchOfflineServer(CommonInitMixIn, OfflineServer):
    def __init__(self, config_dict):
        CommonInitMixIn.__init__(self, config_dict, is_valid=True)
    
    # @override
    # def _server_finalize(self, exit_: int = 0) -> None:
    #     logger.info("Going to merge trajectories now.")
    #     merge_npy_files("$CWD/trajectories/")
    #     logger.info("Merging trajectories done.")

    #     super()._server_finalize(exit_)
    
    # @override
    # def _all_done(self) -> bool:
    #     logger.info((f"Rank {self.rank}>> calls all done."
    #     f"\nNb groups finished: {self.nb_finished_groups} / {self.nb_groups}"
    #     f"\nIs_receiving: {self._is_receiving}"
    #     f"\nGroups: {self._groups}"
    #     f"\nFinished groups: {self.finished_groups}"
    #     f"\nNB Submitted groups: {self.nb_submitted_groups}"))
        
    #     super()._all_done()


class APEBenchServer(CommonInitMixIn,
                     ExperimentalDeepMelissaActiveSamplingServer):

    def __init__(self, config_dict):
        CommonInitMixIn.__init__(self, config_dict)
        if self.world_rank == self.breed_rank:
            return
        self.monitoring_config = config_dict.get("monitoring_config", dict())
        self.__post_init__()

    def __post_init__(self):
        # Setting monitoring config
        self.memory_monitor = mutils.MemoryMonitor(
            mode=self.monitoring_config.get("log_memory", "off"),
            tb_logger=None # self.tb_logger TODO
        )
        self.save_model_freq = self.monitoring_config.get("checkpoint_model_each_step", 0)
        if self.no_fault_tolerance and self.save_model_freq < 100:
            logger.warning((f"Fault tolerance is not active, but model checkpointing is requested every {self.save_model_freq} batch." 
                           "Frequency is changed to 100."))
            self.save_model_freq = 100
        

        self.memory_monitor.log_stats("At init", iteration=0)
        # self.memory_monitor.log_stats("After preparing training attributes", iteration=0)

        # Setting validation data
        self.valid_rollout = self.dl_config.get("valid_rollout", -1)
        self.valid_batch_size = self.dl_config.get("valid_batch_size", 32)
        self.valid_nb_time_steps = self.dl_config.get("valid_nb_time_steps", self.nb_time_steps)
        self.valid_num_samples = self.dl_config.get("valid_num_samples", None)
        self.valid_dataset, self.valid_parameters, self.valid_dataloader, self.valid_dataloader_rollout = dl_utils.load_validation_dataset(
            validation_dir=self.dl_config.get("validation_directory", None),
            validation_file=self.dl_config.get("validation_file", None),
            nb_time_steps=self.valid_nb_time_steps,
            batch_size=self.valid_batch_size,
            rollout_size=self.valid_rollout,
            num_samples=self.valid_num_samples
        )
        # validation rollout is intensive and very slow
        self.one_small_batch = self.dl_config.get("valid_rollout_fast", False)
        self.best_val_loss = np.inf

        # Setting plotting config
        self.plotting_config = self.monitoring_config.get("plotting_config", None)
        # if self.plotting_config is not None:
        #     self.plot_dim = self.scenario.num_spatial_dims
        #     nrows = self.plotting_config.get("plot_rows", 5)
        #     self.plot_row_ids = np.linspace(0, self.valid_batch_size, num=nrows, dtype=int)
        #     self.plot_tids = np.linspace(0, self.valid_nb_time_steps, num=nrows, dtype=int)
    
    def get_learning_rate(self):
        _, scheduler = self.scenario.get_optimizer(with_lr_scheduler=True)
        count = self.opt_state[0].count # get current step
        lr = scheduler(count) # get learning rate from schedule
        return lr.item()
    
    def get_breed_ratio(self, sim_id_list):
        is_bred_list = []
        for sim_id in sim_id_list:
            is_bred_list.append(self._parameter_sampler.current_metadata_list[sim_id].is_bred)
        return np.mean(is_bred_list)

    @override
    def prepare_training_attributes(self):
        model = self.scenario.get_network()
        optimizer = self.scenario.get_optimizer()
        self.opt_state = optimizer.init(eqx.filter(model, eqx.is_array))
        
        logger.info(
            f"TRAINING:00000:\nModel parameters count: {pdeqx.count_parameters(model)}"
            f"\nOptimizer config: {self.scenario.scenario.optim_config}"
            f"\nAPEBench scenario: {self.scenario.scenario}"
        )
        
        return model, optimizer
    
    @override
    def process_simulation_data(self, msg, config_dict):
        u_prev = msg.data["preposition"].reshape(*self.mesh_shape)
        u_next = msg.data["position"].reshape(*self.mesh_shape)
        return u_prev, u_next, msg.simulation_id, msg.time_step
    
    @override
    def on_train_start(self):
        # TODO shitcoded because for example tb_logger is not set in the constructor
        self.memory_monitor.set_tb_logger(self.tb_logger)
        self.tb_logger_layout = {
            "Loss": {
            "Train_Val": ["Multiline", ["Loss/train", "Loss_valid/mean"]],
            "Val_Mean_Min_Max": ["Margin", ["Loss_valid_stats/mean_mean", "Loss_valid_stats/min", "Loss_valid_stats/max"]],
            "Val_Mean_Percentile_10_90": ["Margin", ["Loss_valid_stats/p50", "Loss_valid_stats/p10", "Loss_valid_stats/p90"]],
            "Val_Mean_Percentile_25_75": ["Margin", ["Loss_valid_stats/p50", "Loss_valid_stats/p25", "Loss_valid_stats/p75"]],
            "Val_Mean_pm_1std": ["Margin", ["Loss_valid_stats/mean_mean", "Loss_valid_stats/mean_mstd", "Loss_valid_stats/mean_pstd"]]
            },
            "Rollout": {
                "Both": ["Multiline", [f"Loss_valid/rollout_longer_n{self.valid_rollout}_nRMSE", f"Loss_valid/rollout_seen_n{self.nb_time_steps}_nRMSE"]],
                "Longer": ["Multiline", [f"Loss_valid/rollout_longer_n{self.valid_rollout}_nRMSE"]],
                "Seen": ["Multiline", [f"Loss_valid/rollout_seen_n{self.nb_time_steps}_nRMSE"]]
            },
            "Other": {
                "Breed_ratio": ["Multiline", ["Breed_ratio/batch"]],
                "Learning_rate": ["Multiline", ["Learning_rate"]]
            }
        }
        self.tb_logger_layout.update(self.memory_monitor.tb_logger_layout)
        try:
            self.tb_logger._writer.add_custom_scalars(self.tb_logger_layout)
        except Exception as e:
            logger.error(f"Error adding custom scalars to TensorBoard: {e}")
            logger.error("Custom scalars layout may not be applied correctly.")
        # TODO

        self.memory_monitor.log_stats("After preparing training attributes", iteration=0)
        logger.info("TRAINING:00000: Training will start as soon as watermark is met.")
        
        if (
            self.rank == 0
            and self._valid_dataloader is not None
        ):
            logger.info(f"Rank {self.rank} Running validation at batch_idx={0}")
            start = time.time()
            b_times = []
            self._on_validation_start(0)
            for v_batch_idx, v_batch in enumerate(self._valid_dataloader):
                b_start = time.time()
                self.validation_step(v_batch, v_batch_idx, 0)
                b_times.append(time.time() - b_start)
            main_time = time.time() - start
            self._on_validation_end(0)
            total_time = time.time() - start
            logger.info((f"Validation successfully completed at batch_idx=0. "
                        f"\n\tTotal time: {total_time:.2f} seconds. Main time: {main_time:.2f} seconds. "
                        f"\n\tRollout time: {total_time - main_time:.2f} seconds. "
                        f"\n\tAverage time per batch: {np.mean(b_times):.2f} seconds. Max time per batch: {np.max(b_times):.2f} seconds."))
            self.memory_monitor.log_stats("After validation", iteration=1)

    @override
    def training_step(self, batch, batch_idx):
        u_prev, u_next, sim_ids_list, time_step_list = batch

        with jax.default_device(jax.devices("gpu")[0]):
            u_prev = jnp.asarray(u_prev)
            u_next = jnp.asarray(u_next)
            self.model, self.opt_state, batch_loss, loss_per_sample = dl_utils.update_fn(
                self.model, self.optimizer, u_prev, u_next, self.opt_state
            )
        
        self.memory_monitor.log_stats("Training step", iteration=batch_idx)
        
        self.tb_logger.log_scalar("Loss/train", batch_loss.item(), batch_idx)
        self.tb_logger.log_scalar("Learning_rate", self.get_learning_rate(), batch_idx)
        self.tb_logger.log_scalar("Breed_ratio/batch", self.get_breed_ratio(sim_ids_list), batch_idx)
        
        # TODO these should be logged always, not when log_extra==True :(
        # self.tb_logger.log_scalar("Breed_ratio/R", self._parameter_sampler.R, batch_idx)
        # self.tb_logger.log_scalar("Breed_iter", self._parameter_sampler.R_i, batch_idx)

        if batch_idx % 50 == 0:
            logger.info(f"TRAINING:{batch_idx:05d}: Batch loss: {batch_loss.item():.2e}")

        if not np.isfinite(batch_loss.item()):
            logger.error(f"NaN or Inf loss encountered at batch {batch_idx}.")
            logger.error(f"LOSSES = {loss_per_sample}")
            logger.error(f"SIM IDS = {sim_ids_list}")
            logger.error(f"TIME STEPS = {time_step_list}")
            time.sleep(5)
            os.exit(1)

        delta_losses = \
            active_sampling.calculate_delta_loss(np.asarray(loss_per_sample))
        active_sampling.record_increments(sim_ids_list, time_step_list, delta_losses)
        
        if batch_idx % self.save_model_freq == 0:
            logger.info(f"TRAINING:{batch_idx:05d}: Saving model.")
            self.checkpoint_model(batch_idx, suffix=batch_idx)


    @override
    def on_train_end(self):
        if self.monitoring_config.get("checkpoint_model_last", False):
            logger.info("Saving last model.")
            self.checkpoint_model(suffix="last")
        # fig = putils.plot_seen_count_histogram(list(self.buffer.seen_ctr.elements()))
        # if fig is not None:
        #     self.tb_logger.log_figure(
        #         "SeenCountsHistogram",
        #         fig
        #     )
        # TODO create plot of predictions?
        # TODO create plot of final parameters trained on
        # TODO create loss plots for paper?
        # TODO timing ?

    @override
    def on_validation_start(self, batch_idx):
        """Keep track of some variables for validation loop."""
        if self.buffer.seen_ctr:
                fig, ax = plt.subplots(1, 1, figsize=(10, 5))
                ax.bar(
                    self.buffer.seen_ctr.keys(),
                    self.buffer.seen_ctr.values(),
                )
                ax.set_xlabel("Sample Seen Count (log-scale)")
                ax.set_ylabel("Frequency")
                ax.set_xscale("log")
                self.tb_logger.log_figure(
                    "Seen_count_hist",
                    fig,
                    batch_idx
                )
                plt.close(fig)
        self.losses_per_sample = []
        self.batch_val_losses = []
        self.memory_monitor.log_stats("Before validation start", iteration=batch_idx)
        logger.info(f"TRAINING:{batch_idx:05d}: Validation started.")

    @override
    def validation_step(self, batch, valid_batch_idx, batch_idx):
        """This loss is across all trajectories and their time steps.
        t[i] -> t[i + 1] 
        """
        with jax.default_device(jax.devices("gpu")[0]):
            u_step = jnp.asarray(self.valid_dataset[batch[0]])
            u_step = jax.jit(jax.vmap(self.model), donate_argnums=0)(u_step)
            u_next = jnp.asarray(self.valid_dataset[batch[1]])
            loss_per_sample = jnp.mean((u_step - u_next) ** 2, axis=self.mesh_axes)
            batch_loss = jnp.mean(loss_per_sample)

        if not np.isfinite(batch_loss.item()):
            logger.error(f"NaN or Inf loss encountered at batch {valid_batch_idx}.")
            logger.error(f"LOSSES = {loss_per_sample}")
        
        self.batch_val_losses.append(batch_loss.item())
        self.losses_per_sample.append(loss_per_sample.ravel())
        self.memory_monitor.log_stats(f"After val batch {valid_batch_idx}", iteration=batch_idx)
        
    @override
    def on_validation_end(self, batch_idx):
        avg_val_loss = np.nanmean(self.batch_val_losses) if len(self.batch_val_losses) > 0 else np.nan
        self.tb_logger.log_scalar("Loss_valid/mean", avg_val_loss, batch_idx)

        self.memory_monitor.log_stats("End of validation", iteration=batch_idx)
        logger.info(f"TRAINING:{batch_idx:05d}: Validation ended. Average loss: {avg_val_loss:.2e}")

        self.losses_per_sample = np.hstack(self.losses_per_sample)
        mean_tmp = np.nanmean(self.losses_per_sample)
        std_tmp = np.nanstd(self.losses_per_sample)

        if len(self.losses_per_sample) > 0:
            stats = {
                    "max" : np.nanmax(self.losses_per_sample),
                    "min" : np.nanmin(self.losses_per_sample),
                    "std" : std_tmp,
                    "mean_pstd" : mean_tmp + std_tmp,
                    "mean_mean" : mean_tmp,
                    "mean_mstd" : max(max(mean_tmp - std_tmp, 0), max(mean_tmp - 0.5 * std_tmp, 0)), 
                    "p90" : np.nanpercentile(self.losses_per_sample, 90),
                    "p75" : np.nanpercentile(self.losses_per_sample, 75),
                    "p50" : np.nanpercentile(self.losses_per_sample, 50),
                    "p25" : np.nanpercentile(self.losses_per_sample, 25),
                    "p10" : np.nanpercentile(self.losses_per_sample, 10)
                }
            for key, val in stats.items():
                self.tb_logger.log_scalar(f"Loss_valid_stats/{key}", val, batch_idx)

        if self.monitoring_config.get("checkpoint_model_best", False):
            if avg_val_loss < self.best_val_loss:
                self.best_val_loss = avg_val_loss
                logger.info(f"TRAINING:{batch_idx:05d}: Saving best model.")
                self.checkpoint_model(batch_idx, suffix="best")

                fig, ax = plt.subplots(1, 1)
                sc = ax.scatter(
                    *self.valid_parameters[:,::2].T,
                    c=self.losses_per_sample.reshape(-1, self.valid_dataset.nb_time_steps - 1).mean(-1),
                    cmap='Reds', vmin=0.0, s=100, alpha=0.5
                )
                fig.colorbar(sc)
                ax.set_xlabel("Amplitude 1")
                ax.set_ylabel("Amplitude 2")
                ax.set_title("Validation loss scatter plot")
                self.tb_logger.log_figure(
                    "Scatter/ValidationLoss",
                    fig,
                    batch_idx
                )
                plt.close(fig)
        
        # after running regular validation, run the rollout from ICs
        if self.rank == 0 and self.valid_rollout > 1:
            logger.info(f"TRAINING:{batch_idx:05d}: Running validation rollout n={self.valid_rollout}.")
            start_val = time.time()
            self.run_validation_rollout(
                batch_idx, memory_efficient=True,
                plot_rollout_loss=True, plot_prediction=True, one_small_batch=self.one_small_batch)
            logger.info(f"TRAINING:{batch_idx:05d}: Validation rollout function took {time.time() - start_val:.2f} seconds.")
            # self.validation_mesh_plot(batch_idx, valid_batch_idx, sim_ids_list, u_prev, u_next, u_next_hat)
            self.memory_monitor.log_stats(f"After validation rollout", iteration=batch_idx)
            # when we want to add plots
            # predictions, rollout_loss = self.run_validation_rollout(batch_idx, memory_efficient=False)
            # plot predictions, plot rollout_loss

    def run_validation_rollout(
        self, batch_idx, memory_efficient=True,
        plot_rollout_loss=False, plot_prediction=False, one_small_batch=False
        ):
        """This loss is across all trajectories rolled out from their respective ICs.
        t[0] -> t[1] -> ... t[rollout]
        """

        if one_small_batch:
            batch_size = self.dl_config["batch_size"] // 10

        total = self.valid_dataset.num_samples
        if one_small_batch:
            total = batch_size
        rollout_loss = 0.0
        if self.valid_rollout > self.nb_time_steps:
            rollout_known_loss = 0.0
        if plot_rollout_loss:
            rollout_losses = [0.0 for _ in range(self.valid_rollout)]
        # by batches
        start = time.time()

        for ic_idx, rollout_idx in self.valid_dataloader_rollout:
            if one_small_batch:
                ic_idx = ic_idx[:batch_size]
            u_step = jnp.asarray(self.valid_dataset[ic_idx])
            # we can either use ex.rollout to get full trajectory (meaning, GPU memory should fit nb_time_steps * batch_size * mesh_shape * 2)
            # or we can use the model to predict step by step, which is more memory efficient but maybe slower
            if memory_efficient:
                # def rollout_fn(u_step, roll_step):
                #     """Predict the next step and calculate the loss.
                #     Used for lax.scan and will give last predicted step and array of errors over the rollout"""
                #     u_step = jax.vmap(self.model)(u_step)
                #     err = jax.vmap(ex.metrics.nRMSE)(u_step, roll_step)
                #     return u_step, err
                # rollout_steps = jnp.asarray(self.valid_dataset[rollout_idx])
                # u_step, rollout_loss_batch = jax.lax.scan(rollout_fn, u_step, rollout_steps)

                # in the end, lax scan is maybe efficient for not having every prediction stored
                # but we need to load the whole batch trajectory to gpu for loss calculation
                # so it is not feasible
                rollout_loss = 0.0
                for r, roll_idx in enumerate(rollout_idx, start=1):
                    if one_small_batch:
                        roll_idx = roll_idx[:batch_size]
                    u_step = jax.jit(jax.vmap(self.model), donate_argnums=0)(u_step)
                    u_next = jnp.asarray(self.valid_dataset[roll_idx])
                    loss = jnp.sum(jax.vmap(ex.metrics.nRMSE)(u_step, u_next))
                    rollout_loss += loss.item()
                    if plot_rollout_loss:
                        rollout_losses[r-1] += loss.item()
                    del u_next
                    if self.valid_rollout > self.nb_time_steps and r == self.nb_time_steps:
                        rollout_known_loss = rollout_loss
            else:
                trj_idx = np.vstack(rollout_idx).T
                print(trj_idx.shape)
                print(len(rollout_idx))
                print(len(rollout_idx[0]))
                trj_batch = jnp.asarray(self.valid_dataset[trj_idx])
                u_step = jax.vmap(
                    ex.rollout(self.model, self.valid_dataset.nb_time_steps)
                )(u_step)
                rollout_loss_batch = jax.vmap(jax.vmap(ex.metrics.nRMSE))(
                    u_step, trj_batch
                ) # (B, nb_time_steps)
                rollout_loss += jnp.sum(rollout_loss_batch).item()
                if self.valid_rollout > self.nb_time_steps:
                    rollout_known_loss += jnp.sum(rollout_loss_batch[:,:self.nb_time_steps]).item()
            if one_small_batch:
                break
        logger.info(f"TRAINING:{batch_idx:05d}: Validation rollout took {time.time() - start:.2f} seconds.")

        self.tb_logger.log_scalar(
            f"Loss_valid/rollout_longer_n{self.valid_rollout}_nRMSE",
            rollout_loss / total / self.valid_rollout,
            batch_idx
        )
        logger.info(
            f"TRAINING:{batch_idx:05d}: Validation rollout loss: {rollout_loss / total / self.valid_rollout:.2e}"
        )
        if self.valid_rollout > self.nb_time_steps:
            self.tb_logger.log_scalar(
                f"Loss_valid/rollout_seen_n{self.nb_time_steps}_nRMSE",
                rollout_known_loss / total / self.nb_time_steps,
                batch_idx
            )
            logger.info(
                f"TRAINING:{batch_idx:05d}: Validation known rollout loss: {rollout_known_loss / total / self.nb_time_steps:.2e}"
            )
        if plot_rollout_loss:
            rollout_losses = np.asarray(rollout_losses) / total
            fig, ax = plt.subplots(1,1)
            ax.plot(rollout_losses)
            ax.vlines(
                self.nb_time_steps,
                ymin=0,
                ymax=np.max(rollout_losses),
                color="red",
                linestyle="--",
            )
            ax.grid(which="both")
            ax.set_xlabel("Rollout step")
            ax.set_ylabel("nRMSE")
            ax.set_title(f"Validation rollout loss at batch {batch_idx}")
            self.tb_logger.log_figure(
                f"Loss_valid/rollout_full",
                fig,
                batch_idx
            )
            plt.close(fig)
        if plot_prediction:
            with jax.default_device(jax.devices("gpu")[0]):
                u_step = ex.rollout(self.model, self.valid_dataset.nb_time_steps)(jnp.asarray(self.valid_dataset[0])).squeeze(1).T
            u_true = self.valid_dataset[:self.valid_dataset.nb_time_steps].squeeze(1).T
            fig, ax = plt.subplots(1, 3, figsize=(19,6))
            im1 = ax[0].imshow(u_step, vmin=-2.1, vmax=2.1, cmap='coolwarm', aspect='auto')
            ax[0].set_title("Prediction")
            ax[1].imshow(u_true, vmin=-2.1, vmax=2.1, cmap='coolwarm', aspect='auto')
            ax[1].set_title("True")
            fig.colorbar(im1, ax=ax[0])
            fig.colorbar(im1, ax=ax[1])
            im2 = ax[2].imshow(abs(u_step - u_true), vmin=0, cmap='Reds', aspect='auto')
            ax[2].set_title("Difference")
            fig.colorbar(im2, ax=ax[2])
            fig.tight_layout()
            self.tb_logger.log_figure(
                f"ValidationMeshPredictions",
                fig,
                batch_idx
            )
            plt.close(fig)

        # we can return the predictions and the errors
        # but we don't need them for now
        # return u_step, rollout_loss_batch
        
    # def validation_mesh_plot(self, batch_idx, v_batch_idx, sim_ids, u_prev, u_next, u_next_hat):
    #     # only the first batch
    #     if v_batch_idx == 1:
    #         sim_ids = sim_ids[self.plot_row_ids].tolist()
    #         pids = jnp.asarray(self.plot_row_ids)
    #         tids = jnp.asarray(self.plot_tids)
    #         img = None
    #         nrows = len(self.plot_row_ids)
    #         if self.plot_1d:
    #             ncols = len(self.plot_tids)
    #             meshes = [
    #                 u_prev[pids[:, None], tids],
    #                 u_next[pids[:, None], tids],
    #                 u_next_hat[pids[:, None], tids]
    #             ]
    #             fig = putils.create_subplot_1d(
    #                 nrows,
    #                 ncols,
    #                 self.scenario.domain_extent,
    #                 sim_ids,
    #                 tids,
    #                 meshes
    #             )
    #         elif self.plot_2d:
    #             # extract specific time steps from a
    #             # batch of trajectories
    #             def extract(data):
    #                 return jnp.array([
    #                     data[pid, tid]
    #                     for pid in pids
    #                     for tid in tids
    #                 ])
    #             meshes = [
    #                 extract(data)
    #                 for data in [u_prev, u_next, u_next_hat]
    #             ]
                
    #             fig = putils.create_subplot_2d(
    #                 nrows,
    #                 self.scenario.domain_extent,
    #                 sim_ids,
    #                 tids,
    #                 meshes
    #             )                
    #         if fig is not None:
    #             self.tb_logger.log_figure(
    #                 "ValidationMeshPredictions",
    #                 fig,
    #                 batch_idx
    #             )

    # def validation_loss_scatter_plot(self, batch_idx, loss_by_sim):

    #     if self.valid_parameters is not None:
    #         sids = list(sorted(loss_by_sim.keys()))
    #         ls = [
    #             loss_by_sim[sim_id]
    #             for sim_id in sids
    #         ]
    #         x = self.valid_parameters[sids, 0]
    #         y = self.valid_parameters[sids, 1]
    #         fig = putils.validation_loss_scatter_plot_by_sim(x, y, ls)
    #         self.tb_logger.log_figure(
    #             "Scatter/ValidationLoss",
    #             fig,
    #             batch_idx
    #         )

    @override
    def checkpoint(self, batch_idx, path="checkpoints"):
        pass

    def checkpoint_model(self, batch_idx=0, suffix=None):
        """Save model checkpoint to the specified path."""
        if self.rank == 0:
            if not hasattr(self, "checkpoint_model_path") and suffix is None:
                os.makedirs("checkpoints", exist_ok=True)
                self.checkpoint_model_path = "checkpoints/model.eqx"
                checkpoint_model_path = self.checkpoint_model_path
            elif suffix is not None:
                os.makedirs("checkpoints", exist_ok=True)
                checkpoint_model_path = f"checkpoints/model_{suffix}.eqx"
            
            # Create checkpoint dict with all necessary state
            checkpoint_dict = {
                'model': self.model,
                'optimizer': self.optimizer,
                'opt_state': self.opt_state,
                'batch_idx': batch_idx
            }
            
            # Save using equinox serialization
            eqx.tree_serialise_leaves(checkpoint_model_path, checkpoint_dict)
            logger.info(f"TRAINING:{batch_idx:05d}: Saved model checkpoint to {checkpoint_model_path}")

    @override
    def _load_model_from_checkpoint(self):
        """Load model from the latest checkpoint if available."""
        if not hasattr(self, "checkpoint_model_path"):
            self.checkpoint_model_path = "checkpoints/model.eqx"
        try:
            # Load checkpoint dict
            checkpoint_dict = eqx.tree_deserialise_leaves(self.checkpoint_model_path)
            
            # Restore model state
            self.model = checkpoint_dict['model']
            self.optimizer = checkpoint_dict['optimizer']
            self.opt_state = checkpoint_dict['opt_state']
            self.batch_offset = checkpoint_dict['batch_idx']
            logger.info(f"Loaded checkpoint from {self.checkpoint_model_path}")
         
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}")


    @override
    def _setup_environment_slurm(self):
        pass
