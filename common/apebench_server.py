# flake8: noqa
import logging

from typing_extensions import override
import numpy as np
import jax.numpy as jnp
import torch
import pdequinox as pdeqx

from melissa.server.deep_learning.tensorboard_logger import (  # type: ignore
    TorchTensorboardLogger
)
from melissa.server.offline_server import OfflineServer  # type: ignore
from melissa.server.deep_learning import active_sampling  # type: ignore
from melissa.server.deep_learning.active_sampling.active_sampling_server import (  # type: ignore
    ExperimentalDeepMelissaActiveSamplingServer
)

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


class APEBenchOfflineServer(CommonInitMixIn, OfflineServer):
    def __init__(self, config_dict):
        CommonInitMixIn.__init__(self, config_dict, is_valid=True)


# TODO: Melissa DL interface will change in the future
# purely from the training perspective. Dataset -> Dataloader -> training
class APEBenchServer(CommonInitMixIn,
                     ExperimentalDeepMelissaActiveSamplingServer):

    def __init__(self, config_dict):
        CommonInitMixIn.__init__(self, config_dict)

        self.valid_rollout = self.dl_config.get("valid_rollout", -1)
        valid_batch_size = 25
        self.mesh_shape = self.scenario.get_shape()
        out = vutils.load_validation_data(
            validation_dir=self.dl_config.get("validation_directory"),
            seed=self.seed,
            valid_batch_size=valid_batch_size,
            nb_time_steps=self.nb_time_steps,
            output_shape=self.scenario.get_shape(),
        )
        self.valid_dataset, self.valid_dataloader, self.valid_parameters = out
        self.opt_state = None

        if self.valid_rollout > 1:
            self.valid_dataset_for_rollout = vutils.load_validation_data(
                validation_dir=self.dl_config.get("validation_directory"),
                seed=self.seed,
                valid_batch_size=valid_batch_size,
                nb_time_steps=self.nb_time_steps,
                output_shape=self.scenario.get_shape(),
                only_trajectories_dataset=True
            )

        # 1D u_prev, u_next, and u_next_hat are plotted on the same plot
        self.plot_1d = self.scenario.num_spatial_dims == 1
        if self.plot_1d:
            nrows = 5
            self.plot_row_ids = np.random.randint(0, valid_batch_size, size=nrows)
            self.plot_tids = [0, 10, 20, 70, 90]

        # 2D u_prev, u_next, u_next_hat, and error are plotted on each column
        # where a row contains a unique time step from different simulations
        self.plot_2d = self.scenario.num_spatial_dims == 2
        if self.plot_2d:
            nrows = 5
            ncols = 4
            self.plot_row_ids = np.random.randint(0, valid_batch_size, size=nrows)
            self.plot_tids = [0, 10, 20, 70, 90]
            assert len(self.plot_tids) == len(self.plot_row_ids)

    @override
    def setup_environment(self):
        # initialize tensorboardLogger with torch
        self._tb_logger = TorchTensorboardLogger(
            self.rank,
            disable=not self.dl_config["tensorboard"],
            debug=self.verbose_level >= 3
        )
        # make sure set_parameter_sampler() is called
        active_sampling.set_tb_logger(self.tb_logger)

    def set_train_dataloader(self):
        self.train_dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=0
        )

    def train(self):
        self.set_train_dataloader()
        for batch_id, batch in enumerate(self.train_dataloader):
            self.training_step(batch, batch_id)

            if (batch_id + 1) % self.nb_batches_update == 0:
                self.run_validation(batch_id)

    def training_step(self, batch, batch_id):
        u_prev, u_next, sim_ids_list, time_step_list = batch
        u_prev = jnp.asarray(u_prev)
        u_next = jnp.asarray(u_next)
        (
            self.model,
            self.opt_state,
            batch_loss,
            loss_per_sample,
            grads
        ) = tutils.update_fn(
            self.model,
            self.optimizer,
            u_prev,
            u_next,
            self.opt_state,
        )
        for key, val in tutils.get_grads_stats(grads).items():
            self.tb_logger.log_scalar(f"Gradients/{key}", val, batch_id)
        logger.info(f"BATCH={batch_id} loss={batch_loss:.2e}")
        self.tb_logger.log_scalar("Loss/train", batch_loss.item(), batch_id)

        if self.is_breed_study:
            loss_per_sample = torch.tensor(loss_per_sample.tolist())
            batch_loss = torch.tensor(batch_loss.item())
            batch_loss_relative = \
                active_sampling.get_relative_loss(loss_per_sample)
            for sim_id, t_step, sample_loss in zip(
                sim_ids_list, time_step_list, loss_per_sample
            ):
                active_sampling.record_delta_loss(
                    sim_id.item(),
                    t_step.item(),
                    sample_loss,
                    batch_loss,
                    batch_loss_relative,
                )

            self.periodic_resampling(batch_id)

    def run_validation(self, batch_id):
        if self.valid_dataloader is not None and self.rank == 0:
            self.run_validation_regular(batch_id)

            if self.valid_rollout > 1:
                self.run_validation_rollout(batch_id)

    def run_validation_regular(self, batch_id):
        """This loss is across all trajectories and their time steps.
        t[i] -> t[i + 1] 
        """

        loss_by_sim = {}
        val_loss = 0.0
        count = 0
        for vid, valid_batch_data in enumerate(self.valid_dataloader):
            u_prev, u_next, sim_ids = valid_batch_data
            batch_shape = u_prev.shape
            u_prev = jnp.asarray(u_prev).reshape(-1, *self.mesh_shape)
            u_next = jnp.asarray(u_next).reshape(-1, *self.mesh_shape)

            batch_loss, loss_per_sample, u_next_hat = tutils.loss_fn(
                self.model,
                u_prev,
                u_next,
                is_valid=True
            )

            loss_by_sim.update({
                s.item(): l
                for s, l in zip(sim_ids, loss_per_sample)
            })
            val_loss += batch_loss.item()
            count += 1
            if (
                (self.plot_1d or self.plot_2d)
                and (batch_id + 1) % (2 * self.nb_batches_update) == 0
            ):
                u_prev = u_prev.reshape(*batch_shape)
                u_next = u_next.reshape(*batch_shape)
                u_next_hat = u_next_hat.reshape(*batch_shape)
                self.validation_mesh_plot(
                    batch_id, vid, sim_ids, u_prev, u_next, u_next_hat
                )
        # endfor
        avg_val_loss = val_loss / count if count > 0 else 0.0
        self.tb_logger.log_scalar("Loss/valid", avg_val_loss, batch_id)        
        # self.validation_loss_scatter_plot(batch_id, loss_by_sim)

    def run_validation_rollout(self, batch_id):
        """This loss is across all trajectories rolled out from their respective ICs.
        t[0] -> t[1] -> ... t[rollout]
        """

        total = len(self.valid_dataset_for_rollout)
        all_trajectories, _ = self.valid_dataset_for_rollout[[i
            for i in range(total)
        ]]
        all_trajectories = jnp.asarray(all_trajectories)
        mean_loss, _, _ = tutils.rollout_loss_fn(
            self.model,
            all_trajectories,
            self.valid_rollout
        )
        self.tb_logger.log_scalar(
            f"Loss/valid_rollout (n={self.valid_rollout}) nRMSE",
            mean_loss.item(),
            batch_id
        )

    @override
    def prepare_training_attributes(self):

        model = self.scenario.get_network()
        optimizer = self.scenario.get_optimizer()
        logger.info(f"Model parameters count: {pdeqx.count_parameters(model)}")

        return model, optimizer

    @override
    def process_simulation_data(self, msg, config_dict):
        u_prev = msg.data["preposition"]
        u_next = msg.data["position"]

        u_prev = u_prev.reshape(*self.mesh_shape)
        u_next = u_next.reshape(*self.mesh_shape)

        u_prev = np.array(u_prev, copy=True)
        u_next = np.array(u_next, copy=True)

        return u_prev, u_next, msg.simulation_id, msg.time_step

    def validation_mesh_plot(self, batch_id, vid, sim_ids, u_prev, u_next, u_next_hat):
        # only the first batch
        if vid == 0:
            sim_ids = sim_ids[self.plot_row_ids].tolist()
            pids = jnp.asarray(self.plot_row_ids)
            tids = jnp.asarray(self.plot_tids)
            img = None
            nrows = len(self.plot_row_ids)
            if self.plot_1d:
                ncols = len(self.plot_tids)
                meshes = [
                    u_prev[pids[:, None], tids],
                    u_next[pids[:, None], tids],
                    u_next_hat[pids[:, None], tids]
                ]
                img = putils.create_subplot_1d(
                    nrows,
                    ncols,
                    self.scenario.domain_extent,
                    sim_ids,
                    tids,
                    meshes
                )
            elif self.plot_2d:
                # extract specific time steps from a
                # batch of trajectories
                def extract(data):
                    return jnp.array([
                        data[pid, tid]
                        for pid in pids
                        for tid in tids
                    ])
                meshes = [
                    extract(data)
                    for data in [u_prev, u_next, u_next_hat]
                ]
                
                img = putils.create_subplot_2d(
                    nrows,
                    self.scenario.domain_extent,
                    sim_ids,
                    tids,
                    meshes
                )                
            if img is not None:
                self.tb_logger.writer.add_image(
                    "ValidationMeshPredictions",
                    img,
                    batch_id,
                    dataformats="HWC",
                )

    def validation_loss_scatter_plot(self, batch_id, loss_by_sim):

        if self.valid_parameters is not None:
            sids = list(sorted(loss_by_sim.keys()))
            ls = [
                loss_by_sim[sim_id]
                for sim_id in sids
            ]
            x = self.valid_parameters[sids, 0]
            y = self.valid_parameters[sids, 1]
            img = putils.validation_loss_scatter_plot_by_sim(x, y, ls)
            self.tb_logger.writer.add_image(
                "Scatter/ValidationLoss",
                img,
                batch_id,
                dataformats="HWC"
            )

    @override
    def checkpoint(self):
        pass

    @override
    def _load_model_from_checkpoint(self):
        pass

    @override
    def _setup_environment_slurm(self):
        pass
