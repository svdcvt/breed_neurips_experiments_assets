# flake8: noqa
import os
import time
import logging

from typing_extensions import override
import numpy as np
import jax.numpy as jnp
import pdequinox as pdeqx

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
from sampler import get_sampler_class_type
from scenarios import MelissaSpecificScenario

logger = logging.getLogger("melissa")
logging.getLogger("jax").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)


class CommonInitMixIn:
    def __init__(self, config_dict, is_valid=False):
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

        self.valid_rollout = self.dl_config.get("valid_rollout", -1)
        self.valid_batch_size = self.dl_config.get("valid_batch_size", 25)
        self.mesh_shape = self.scenario.get_shape()
        logger.info(f"Mesh shape: {self.mesh_shape}")
        logger.info("Validation configuration starts loading.")
        self.valid_dataset, self.valid_parameters = vutils.load_validation_dataset(
            validation_dir=self.dl_config.get("validation_directory"),
            nb_time_steps=101, # TODO this is number of time steps for validation not for training
            output_shape=self.mesh_shape,
        )
        self.valid_dataloader = vutils.batch_generator(self.valid_dataset, self.valid_batch_size)
        self.opt_state = None

        # 1D u_prev, u_next, and u_next_hat are plotted on the same plot
        self.plot_1d = self.scenario.num_spatial_dims == 1
        if self.plot_1d:
            nrows = 5
            self.plot_row_ids = np.random.randint(0,  self.valid_batch_size, size=nrows)
            self.plot_tids = [0, 10, 20, 70, 90]

        # 2D u_prev, u_next, u_next_hat, and error are plotted on each column
        # where a row contains a unique time step from different simulations
        self.plot_2d = self.scenario.num_spatial_dims == 2
        if self.plot_2d:
            nrows = 5
            ncols = 4
            self.plot_row_ids = np.random.randint(0, self.valid_batch_size, size=nrows)
            self.plot_tids = [0, 10, 20, 70, 90]
            assert len(self.plot_tids) == len(self.plot_row_ids)

        # we can make this optional
        self.log_extra = config_dict["study_options"].get("log_extra", False)
        if self.log_extra:
            self.plot_loss_distributions = dict(
                train=putils.DynamicHistogram(title='Batch loss Distribution', cmap='Blues', show_last=50),
                validation=putils.DynamicHistogram(title='Validation loss Distribution', show_last=50))
            if self.valid_rollout > 1:
                self.plot_loss_distributions.update(
                    valid_rollout=putils.DynamicHistogram(
                        title='Validation rollout loss Distribution',
                        show_last=50
                    )
                )

    @override
    def prepare_training_attributes(self):

        model = self.scenario.get_network()
        optimizer = self.scenario.get_optimizer()
        logger.info(f"Model parameters count: {pdeqx.count_parameters(model)}")
        
        if self.opt_state is None:
            self.opt_state = tutils.init_optimizer_state(optimizer, model)
        logger.info("Training attributes prepared.")
        return model, optimizer

    @override
    def on_train_end(self):
        fig = putils.plot_seen_count_histogram(list(self.buffer.seen_ctr.elements()))
        if fig is not None:
            self.tb_logger.log_figure(
                "SeenCountsHistogram",
                fig
            )

    @override
    def training_step(self, batch, batch_idx):
        u_prev, u_next, sim_ids_list, time_step_list = batch
        u_prev = jnp.asarray(u_prev)
        u_next = jnp.asarray(u_next)
        (
            self.model,
            self.opt_state,
            batch_loss,
            loss_per_sample
        ) = tutils.update_fn(
            self.model,
            self.optimizer,
            u_prev,
            u_next,
            self.opt_state
        )
        if jnp.isnan(batch_loss):
            logger.error(f"LOSSES = {loss_per_sample}")
            logger.error(f"SIM IDS = {sim_ids_list}")
            logger.error(f"TIME STEPS = {time_step_list}")
            time.sleep(5)
            os.exit(1)
        
        self.tb_logger.log_scalar("Loss/train", batch_loss.item(), batch_idx)

        if self.log_extra:
            self.plot_loss_distributions['train'].add_histogram_step(
                np.asarray(loss_per_sample)
            )
            self.tb_logger.log_figure(
                "TrainLossHistogram",
                self.plot_loss_distributions['train'].fig,
                batch_idx,
                close=False
            )

        if self.is_breed_study:
            delta_losses = \
                active_sampling.calculate_delta_loss(np.asarray(loss_per_sample))
            active_sampling.record_increments(sim_ids_list, time_step_list, delta_losses)
        
        jax.clear_caches()

    @override
    def on_validation_start(self, batch_idx):
        """Keep track of some variables for validation loop."""
        self.loss_by_sim = {}
        self.losses_per_sample = []
        self.batch_val_losses = []
        # TODO shitcoded generator because it was empty after previous validation lol
        self.valid_dataloader = vutils.batch_generator(self.valid_dataset, self.valid_batch_size)
    @override
    def validation_step(self, batch, valid_batch_idx, batch_idx):
        """This loss is across all trajectories and their time steps.
        t[i] -> t[i + 1] 
        """
        u_prev_indices, u_next_indices = batch
        u_step = jnp.asarray(self.valid_dataset[u_prev_indices]).reshape(-1, *self.mesh_shape)
        u_step = jax.jit(jax.vmap(self.model), donate_argnums=0)(u_step)
        with jax.default_device(jax.devices("cpu")[0]):
            u_next = jnp.asarray(self.valid_dataset[u_next_indices]).reshape(-1, *self.mesh_shape)
            loss_per_sample = jnp.mean((u_step - u_next) ** 2, axis=tuple(range(1, len(self.mesh_shape) + 1)))
            batch_loss = jnp.mean(loss_per_sample)
            if batch_loss is np.nan:
                logger.error(f"LOSSES = {loss_per_sample}")
            self.losses_per_sample.append(
                loss_per_sample.ravel()
            )
            self.batch_val_losses.append(batch_loss.item())
    @override
    def on_validation_end(self, batch_idx):
        avg_val_loss = np.nanmean(self.batch_val_losses) if len(self.batch_val_losses) > 0 else np.nan
        self.tb_logger.log_scalar("Loss_valid/mean", avg_val_loss, batch_idx)

        self.losses_per_sample = np.hstack(self.losses_per_sample)
        logger.info(f"Nan count: {np.isnan(self.losses_per_sample).sum()}")

        if len(self.losses_per_sample) > 0:
            self.tb_logger.log_scalar("Loss_valid/max", 
                                      np.nanmax(self.losses_per_sample),
                                      batch_idx)
            self.tb_logger.log_scalar("Loss_valid/min",
                                      np.nanmin(self.losses_per_sample),
                                      batch_idx)
            self.tb_logger.log_scalar("Loss_valid/p90",
                                      np.nanpercentile(self.losses_per_sample, 90),
                                      batch_idx)
            self.tb_logger.log_scalar("Loss_valid/p10",
                                      np.nanpercentile(self.losses_per_sample, 10),
                                      batch_idx)

        if self.log_extra:
            self.plot_loss_distributions['validation'].add_histogram_step(
                self.losses_per_sample
            )
            self.tb_logger.log_figure(
                "ValidationLossHistogram",
                self.plot_loss_distributions['validation'].fig,
                batch_idx,
                close=False
            )

        # after running regular validation, run the rollout from ICs
        if self.rank == 0 and self.valid_rollout > 0:
            logger.info("Running validation rollout.")
            self.run_validation_rollout(batch_idx, save_intermediate=True)
        jax.clear_caches()
        self.memory_monitor.print_stats(f"After clearing cache in validation batch {batch_idx}")


    def run_validation_rollout(self, batch_idx, save_intermediate=False):
        """This loss is across all trajectories rolled out from their respective ICs.
        t[0] -> t[1] -> ... t[rollout]
        """
        total = self.valid_dataset.num_samples
        if save_intermediate:
            rollout_losses = []
            for b, (ic_idx, rollout_idx) in enumerate(vutils.batch_generator(self.valid_dataset, self.valid_batch_size, rollout_size=self.valid_rollout)):
                u_step = jnp.asarray(self.valid_dataset[ic_idx]).reshape(-1, *self.mesh_shape)
                batch_rollout_losses = []
                for r, roll_idx in enumerate(rollout_idx):
                    u_step = jax.jit(jax.vmap(self.model), donate_argnums=0)(u_step)
                    with jax.default_device(jax.devices("cpu")[0]):
                        u_next = jnp.asarray(self.valid_dataset[roll_idx]).reshape(-1, *self.mesh_shape)
                        batch_rollout_losses.append(jnp.mean((u_step - u_next) ** 2, axis=tuple(range(1, len(self.mesh_shape) + 1))).ravel())
                rollout_losses.append(np.vstack(batch_rollout_losses).T)
            rollout_losses = np.vstack(rollout_losses)
            logger.info(f"Nan count: {np.isnan(rollout_losses).sum()}")
            total_loss = np.nansum(rollout_losses[:,:self.valid_rollout])
            logger.info(f"Total rollout loss: {total_loss:.2e}")
            self.tb_logger.log_scalar(
                f"Loss_valid/rollout (n={self.valid_rollout}) nRMSE",
                (total_loss / total).item(),
                batch_idx
            )
            if self.valid_rollout > self.nb_time_steps:
                total_nb_loss = np.nansum(rollout_losses[:,:self.nb_time_steps])
                self.tb_logger.log_scalar(
                    f"Loss_valid/rollout (n={self.nb_time_steps}) nRMSE",
                    (total_nb_loss / total).item(),
                    batch_idx
                )
        else:
            total_nrmse = 0.0
            for ic_idx, rollout_idx in vutils.batch_generator(self.valid_dataset, self.valid_batch_size, rollout_size=self.valid_rollout):
                ics = jnp.asarray(self.valid_dataset[ic_idx]).reshape(-1, *self.mesh_shape)
                y_pred = jax.jit(jax.vmap(ex.rollout(self.model, self.valid_rollout, include_init=False)), donate_argnums=0)(ics)
                with jax.default_device(jax.devices("cpu")[0]):
                    rolled_y = jnp.asarray(self.valid_dataset[rollout_idx[-1]]).reshape(-1, *self.mesh_shape)
                    total_nrmse += jnp.sum(jax.vmap(
                        ex.metrics.nRMSE,
                        in_axes=1 # not sure for 2D
                    )(y_pred, rolled_y))
            
            self.tb_logger.log_scalar(
                f"Loss_valid/rollout (n={self.valid_rollout}) nRMSE",
                (total_nrmse / total).item(),
            )
    
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
