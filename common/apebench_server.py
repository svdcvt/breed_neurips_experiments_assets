# flake8: noqa
import os
import time
import logging
import gc

from typing_extensions import override
import numpy as np
import jax.numpy as jnp
import jax
import pdequinox as pdeqx
import exponax as ex

from melissa.server.offline_server import OfflineServer  # type: ignore
from melissa.server.deep_learning import active_sampling  # type: ignore
from melissa.server.deep_learning.active_sampling.active_sampling_server import (  # type: ignore
    ExperimentalDeepMelissaActiveSamplingServer
)
# when we merge
# from melissa.utility.plots import DynamicHistogram

import train_utils as tutils
import valid_utils as vutils
import plot_utils as putils
import debug_utils as dutils

from sampler import get_sampler_class_type
from scenarios import MelissaSpecificScenario

logger = logging.getLogger("melissa")
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)


# def debug_print(a, name):
#     if isinstance(a, jnp.ndarray):
#         logger.info(f"LOOK! {name}: device: {a.device} type: {type(a)} dtype: {a.dtype} nbytes: {a.nbytes/1024/1024:.2f} MB | ptr: {a.unsafe_buffer_pointer()}")
#     elif isinstance(a, np.ndarray):
#         logger.info(f"LOOK! {name}: type: {type(a)} dtype: {a.dtype} nbytes: {a.nbytes/1024/1024:.2f} MB | ptr: {a.data} and {np.byte_bounds(a)[0]}")
#     else:
#         logger.info(f"LOOK! {name}: type: {type(a)}")

class CommonInitMixIn:
    def __init__(self, config_dict, is_valid=False):
        self.memory_tracer = dutils.MemoryTracer(off=config_dict['dl_config'].get('mem_monitoring_off', False))
        self.memory_tracer.take_snapshot("At init")

        super().__init__(config_dict)
        study_options = config_dict["study_options"]
        scenario_config = study_options["scenario_config"]
        self.seed = study_options["seed"]
        l_bounds = study_options["l_bounds"]
        u_bounds = study_options["u_bounds"]

        # eval() for string type bounds from json
        for i in range(len(l_bounds)):
            if isinstance(l_bounds[i], str):
                l_bounds[i] = eval(l_bounds[i])
            if isinstance(u_bounds[i], str):
                u_bounds[i] = eval(u_bounds[i])

        sampler_type = config_dict.get("sampler_type", "uniform")
        self.is_breed_study = sampler_type == "breed"
        if self.is_breed_study:
            self.breed_params = self.ac_config.get("breed_params", dict())
        else:
            self.breed_params = {}
        self.scenario = MelissaSpecificScenario(**scenario_config)
        # if not is_valid:
        #     assert self.scenario.train_temporal_horizon == self.nb_time_steps, \
        #         f"This scenario in the benchmark has {self.scenario.train_temporal_horizon:i} time steps in trajectory, but in in configuration file it is set to {self.nb_time_steps:i}"
        # else:
        #     assert self.scenario.test_temporal_horizon == self.nb_time_steps, \
        #         f"This scenario in the benchmark has {self.scenario.test_temporal_horizon:i} time steps in trajectory, but in in configuration file it is set to {self.nb_time_steps:i}"
        ic_config = scenario_config["ic_config"]
        ic_type = ic_config.split(";")[0]
        self.sampler_t = get_sampler_class_type(
            ic_type=ic_type,
            is_breed=self.is_breed_study
        )
        self.set_parameter_sampler(
            sampler_t=self.sampler_t,
            ic_config=ic_config,
            is_valid=is_valid,
            # scenario=self.scenario,
            **self.breed_params,
            seed=self.seed,
            l_bounds=l_bounds,
            u_bounds=u_bounds,
            dtype=np.float32
        )
        self.experimental_monitoring = True


class APEBenchOfflineServer(CommonInitMixIn, OfflineServer):
    def __init__(self, config_dict):
        CommonInitMixIn.__init__(self, config_dict, is_valid=True)


class APEBenchServer(CommonInitMixIn,
                     ExperimentalDeepMelissaActiveSamplingServer):

    def __init__(self, config_dict):
        CommonInitMixIn.__init__(self, config_dict)

        self.memory_monitor = dutils.MemoryMonitor()#off=self.dl_config.get('mem_monitoring_off', False))
        self.memory_monitor.print_stats("At init", with_timestamp=True)

        self.valid_rollout = self.dl_config.get("valid_rollout", -1)
        self.valid_batch_size = self.dl_config.get("valid_batch_size", 25)
        self.mesh_shape = self.scenario.get_shape()
        logger.info(f"Mesh shape: {self.mesh_shape}")
        logger.info("Validation configuration starts loading.")
        self.valid_dataset, self.valid_parameters, self.valid_dataloader, self.valid_dataloader_rollout = vutils.load_validation_dataset(
            validation_dir=self.dl_config.get("validation_directory"),
            nb_time_steps=101, # TODO this is number of time steps for validation not for training
            batch_size=self.valid_batch_size,
            rollout_size=self.valid_rollout
        )

        self.opt_state = None
        self.clear_freq_tr = self.dl_config.get("clear_freq", (50, 1))[0]
        self.clear_freq_val = self.dl_config.get("clear_freq", (50, 1))[1]

        # # 1D u_prev, u_next, and u_next_hat are plotted on the same plot
        # self.plot_1d = self.scenario.num_spatial_dims == 1
        # if self.plot_1d:
        #     nrows = 5
        #     self.plot_row_ids = np.random.randint(0,  self.valid_batch_size, size=nrows)
        #     self.plot_tids = [0, 10, 20, 70, 90]

        # # 2D u_prev, u_next, u_next_hat, and error are plotted on each column
        # # where a row contains a unique time step from different simulations
        # self.plot_2d = self.scenario.num_spatial_dims == 2
        # if self.plot_2d:
        #     nrows = 5
        #     ncols = 4
        #     self.plot_row_ids = np.random.randint(0, self.valid_batch_size, size=nrows)
        #     self.plot_tids = [0, 10, 20, 70, 90]
        #     assert len(self.plot_tids) == len(self.plot_row_ids)

        # we can make this optional
        self.log_extra = config_dict["study_options"].get("log_extra", False)
        # if self.log_extra:
        #     self.plot_loss_distributions = dict(
        #         train=putils.DynamicHistogram(title='Batch loss Distribution', cmap='Blues', show_last=50),
        #         validation=putils.DynamicHistogram(title='Validation loss Distribution', show_last=50))
        #     if self.valid_rollout > 1:
        #         self.plot_loss_distributions.update(
        #             valid_rollout=putils.DynamicHistogram(
        #                 title='Validation rollout loss Distribution',
        #                 show_last=50
        #             )
        #         )
        
        # self.pid = os.getpid()
        # logger.info(f"Server Process ID: {self.pid}")
        self.memory_monitor.print_stats("End of init", with_timestamp=True)

    @override
    def prepare_training_attributes(self):

        model = self.scenario.get_network()
        optimizer = self.scenario.get_optimizer()
        logger.info(f"Model parameters count: {pdeqx.count_parameters(model)}")
        
        if self.opt_state is None:
            self.opt_state = tutils.init_optimizer_state(optimizer, model)
        logger.info("Training attributes prepared.")
        self.memory_monitor.print_stats("After preparing training attributes", with_timestamp=True)
        return model, optimizer
    
    @override
    def process_simulation_data(self, msg, config_dict):
        self.memory_monitor.print_stats(f"Before processing simulation data {msg.simulation_id}-{msg.time_step}", with_timestamp=True)
        u_prev = msg.data["preposition"].reshape(*self.mesh_shape)
        u_next = msg.data["position"].reshape(*self.mesh_shape)
        self.memory_monitor.print_stats("After processing", with_timestamp=True)
        # try:
        #     debug_print(u_prev, f"u_prev j_t={msg.simulation_id}, {msg.time_step} from msg")
        # except Exception as e:
        #     pass

        # u_prev = u_prev.
        # u_next = u_next.

        # try:
        #     debug_print(u_prev, f"u_prev j_t={msg.simulation_id}, {msg.time_step} from msg after reshape")
        # except Exception as e:
        #     pass

        # u_prev = np.array(u_prev, copy=True)
        # u_next = np.array(u_next, copy=True)

        # try:
        #     debug_print(u_prev, f"u_prev j_t={msg.simulation_id}, {msg.time_step} from msg after copy")
        # except Exception as e:
        #     pass

        return u_prev, u_next, msg.simulation_id, msg.time_step
    
    @override
    def on_train_start(self):
        logger.info("Training started.")

    @override
    def training_step(self, batch, batch_idx):
        # logger.info(f"Training batch {batch_idx} started.")
        self.memory_monitor.print_stats("Start of training step", batch_idx, with_timestamp=True)
        u_prev, u_next, sim_ids_list, time_step_list = batch

        device = jax.devices("gpu")[0]
        with jax.default_device(device):
            u_prev = jnp.asarray(u_prev)
            u_next = jnp.asarray(u_next)
            self.model, self.opt_state, batch_loss = tutils.update_fn(
                self.model, self.optimizer, u_prev, u_next, self.opt_state
            )
            batch_loss.block_until_ready()
        
        self.tb_logger.log_scalar("Loss/train", batch_loss.item(), batch_idx)
        # logger.info(f"Batch {batch_idx} Loss: {batch_loss.item():.2e}")

        if (batch_idx + 1) % self.clear_freq_tr == 0:
            self.memory_monitor.print_stats("Before clearing cache in training", batch_idx, with_timestamp=True)
            jax.clear_caches()
            self.memory_monitor.print_stats("After clearing cache", with_timestamp=True)
            gc.collect()
            self.memory_monitor.print_stats(f"After garbage collection in training", with_timestamp=True)
        
        if (batch_idx + 1) % 100 == 0:
            self.memory_tracer.take_snapshot(f"At batch {batch_idx}")
        

        # if jnp.isnan(batch_loss):
        #     logger.error(f"LOSSES = {loss_per_sample}")
        #     logger.error(f"SIM IDS = {sim_ids_list}")
        #     logger.error(f"TIME STEPS = {time_step_list}")
        #     time.sleep(5)
        #     os.exit(1)
        
        # if self.log_extra:
        #     self.plot_loss_distributions['train'].add_histogram_step(
        #         np.asarray(loss_per_sample)
        #     )
        #     self.tb_logger.log_figure(
        #         "TrainLossHistogram",
        #         self.plot_loss_distributions['train'].fig,
        #         batch_idx,
        #         close=False
        #     )

        # if self.is_breed_study:
        #     delta_losses = \
        #         active_sampling.calculate_delta_loss(np.asarray(loss_per_sample))
        #     active_sampling.record_increments(sim_ids_list, time_step_list, delta_losses)

    # @override
    # def on_train_end(self):
    #     fig = putils.plot_seen_count_histogram(list(self.buffer.seen_ctr.elements()))
    #     if fig is not None:
    #         self.tb_logger.log_figure(
    #             "SeenCountsHistogram",
    #             fig
    #         )
    #     # TODO create plot of predictions?
    #     # TODO create plot of final parameters trained on
    #     # TODO create loss plots for paper?
    #     # TODO timing ?

    @override
    def on_validation_start(self, batch_idx):
        """Keep track of some variables for validation loop."""
        # self.losses_per_sample = []
        self.batch_val_losses = []
        self.memory_monitor.print_stats("Before validation start", batch_idx, with_timestamp=True)
        logger.info("Validation started.")

    @override
    def validation_step(self, batch, valid_batch_idx, batch_idx):
        """This loss is across all trajectories and their time steps.
        t[i] -> t[i + 1] 
        """
        with jax.default_device(jax.devices("gpu")[0]):
            u_step = jnp.asarray(self.valid_dataset[batch[0]])

            self.memory_monitor.print_stats(f"After loading val batch {valid_batch_idx}", with_timestamp=True)
            u_step = jax.jit(jax.vmap(self.model), donate_argnums=0)(u_step)

            u_step.block_until_ready()
            self.memory_monitor.print_stats(f"After passing to NN", with_timestamp=True)

            u_next = jnp.asarray(self.valid_dataset[batch[1]])
            batch_loss = jnp.mean((u_step - u_next) ** 2)
            self.batch_val_losses.append(batch_loss.item())
            
            # loss_per_sample = jnp.mean((u_step - u_next) ** 2, axis=tuple(range(1, len(self.mesh_shape) + 1)))
            # batch_loss = jnp.mean(loss_per_sample)
            # if batch_loss is np.nan:
                # logger.error(f"LOSSES = {loss_per_sample}")

            # logger.info(f"Validation batch {valid_batch_idx} Loss: {batch_loss.item():.2e}")
            # self.losses_per_sample.append(
                # loss_per_sample.ravel()
            # )
            del u_step
            del u_next
            del batch_loss
        self.memory_monitor.print_stats(f"After val batch {valid_batch_idx}", with_timestamp=True)
        
    @override
    def on_validation_end(self, batch_idx):
        logger.info("Validation ended.")
        self.memory_monitor.print_stats("End of validation", batch_idx, with_timestamp=True)

        avg_val_loss = np.nanmean(self.batch_val_losses) if len(self.batch_val_losses) > 0 else np.nan
        self.tb_logger.log_scalar("Loss_valid/mean", avg_val_loss, batch_idx)

        gc.collect()
        self.memory_monitor.print_stats(f"After garbage collection at end of validation", batch_idx, with_timestamp=True)


        # self.losses_per_sample = np.hstack(self.losses_per_sample)
        # logger.info(f"Nan count: {np.isnan(self.losses_per_sample).sum()}")

        # if len(self.losses_per_sample) > 0:
        #     self.tb_logger.log_scalar("Loss_valid/max", 
        #                               np.nanmax(self.losses_per_sample),
        #                               batch_idx)
        #     self.tb_logger.log_scalar("Loss_valid/min",
        #                               np.nanmin(self.losses_per_sample),
        #                               batch_idx)
        #     self.tb_logger.log_scalar("Loss_valid/p90",
        #                               np.nanpercentile(self.losses_per_sample, 90),
        #                               batch_idx)
        #     self.tb_logger.log_scalar("Loss_valid/p10",
        #                               np.nanpercentile(self.losses_per_sample, 10),
        #                               batch_idx)

        # if self.log_extra:
        #     self.plot_loss_distributions['validation'].add_histogram_step(
        #         self.losses_per_sample
        #     )
        #     self.tb_logger.log_figure(
        #         "ValidationLossHistogram",
        #         self.plot_loss_distributions['validation'].fig,
        #         batch_idx,
        #         close=False
        #     )
        
        # logger.info(f"Validation loss: {avg_val_loss:.2e}")
        # after running regular validation, run the rollout from ICs

        # if self.rank == 0 and self.valid_rollout > 0:
        #     logger.info("Running validation rollout.")
        #     self.run_validation_rollout(batch_idx, save_intermediate=True)
        #     self.memory_monitor.print_stats(f"After validation rollout")
        
        
        
    # def run_validation_rollout(self, batch_idx, save_intermediate=False):
    #     """This loss is across all trajectories rolled out from their respective ICs.
    #     t[0] -> t[1] -> ... t[rollout]
    #     """
    #     total = self.valid_dataset.num_samples
    #     if save_intermediate:
    #         rollout_losses = []
    #         for b, (ic_idx, rollout_idx) in enumerate(vutils.batch_generator(self.valid_dataset, self.valid_batch_size, rollout_size=self.valid_rollout)):
    #             u_step = jnp.asarray(self.valid_dataset[ic_idx]).reshape(-1, *self.mesh_shape)
    #             batch_rollout_losses = []
    #             for r, roll_idx in enumerate(rollout_idx):
    #                 u_step = jax.jit(jax.vmap(self.model), donate_argnums=0)(u_step)
    #                 with jax.default_device(jax.devices("cpu")[0]):
    #                     u_next = jnp.asarray(self.valid_dataset[roll_idx]).reshape(-1, *self.mesh_shape)
    #                     batch_rollout_losses.append(jnp.mean((u_step - u_next) ** 2, axis=tuple(range(1, len(self.mesh_shape) + 1))).ravel())
    #             rollout_losses.append(np.vstack(batch_rollout_losses).T)
    #         rollout_losses = np.vstack(rollout_losses)
    #         logger.info(f"Nan count: {np.isnan(rollout_losses).sum()}")
    #         total_loss = np.nansum(rollout_losses[:,:self.valid_rollout])
    #         logger.info(f"Total rollout loss: {total_loss:.2e}")
    #         self.tb_logger.log_scalar(
    #             f"Loss_valid/rollout (n={self.valid_rollout}) nRMSE",
    #             (total_loss / total).item(),
    #             batch_idx
    #         )
    #         if self.valid_rollout > self.nb_time_steps:
    #             total_nb_loss = np.nansum(rollout_losses[:,:self.nb_time_steps])
    #             self.tb_logger.log_scalar(
    #                 f"Loss_valid/rollout (n={self.nb_time_steps}) nRMSE",
    #                 (total_nb_loss / total).item(),
    #                 batch_idx
    #             )
    #     else:
    #         total_nrmse = 0.0
    #         for ic_idx, rollout_idx in vutils.batch_generator(self.valid_dataset, self.valid_batch_size, rollout_size=self.valid_rollout):
    #             ics = jnp.asarray(self.valid_dataset[ic_idx]).reshape(-1, *self.mesh_shape)
    #             y_pred = jax.jit(jax.vmap(ex.rollout(self.model, self.valid_rollout, include_init=False)), donate_argnums=0)(ics)
    #             with jax.default_device(jax.devices("cpu")[0]):
    #                 rolled_y = jnp.asarray(self.valid_dataset[rollout_idx[-1]]).reshape(-1, *self.mesh_shape)
    #                 total_nrmse += jnp.sum(jax.vmap(
    #                     ex.metrics.nRMSE,
    #                     in_axes=1 # not sure for 2D
    #                 )(y_pred, rolled_y))
            
    #         self.tb_logger.log_scalar(
    #             f"Loss_valid/rollout (n={self.valid_rollout}) nRMSE",
    #             (total_nrmse / total).item(),
    #         )
    
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
    def checkpoint(self, batch_idx, path):
        pass

    @override
    def _load_model_from_checkpoint(self):
        pass

    @override
    def _setup_environment_slurm(self):
        pass
