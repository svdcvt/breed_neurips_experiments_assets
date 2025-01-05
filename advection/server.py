import logging

from typing_extensions import override
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pdequinox as pdeqx
import torch

from melissa.server.deep_learning import active_sampling

import train_utils as tutils
import valid_utils as vutils
from sampler import CustomICBreeder, CustomICUniformSampler
from apebench_server import BaseAPEBenchServer


logger = logging.getLogger("melissa")


class AdvectionServer(BaseAPEBenchServer):

    def __init__(self, config_dict):
        super().__init__(config_dict)
        study_options = config_dict["study_options"]

        self.advection_config = study_options["advection"]
        self.mesh_shape = [
            self.advection_config["nb_points"]
        ] * self.advection_config["nb_dims"]

        self.seed = study_options["seed"]
        # amplitude, phase
        self.l_bounds = [-1.0, 0.0]
        self.u_bounds = [1.0, 2 * np.pi]

        self.sampler_type = config_dict.get("sampler_type", "uniform")
        if self.sampler_type == "breed":
            self.breed_params = self.ac_config.get("breed_params", dict())
            sampler_t = CustomICBreeder
        else:
            self.breed_params = {}
            sampler_t = CustomICUniformSampler

        self.set_parameter_sampler(
            sampler_t=sampler_t,
            **self.breed_params,
            seed=self.seed,
            l_bounds=self.l_bounds,
            u_bounds=self.u_bounds,
            dtype=np.float32
        )

        self.opt_state = None

        self.valid_dataset, self.valid_dataloader, self.valid_parameters = vutils.load_validation_data(
            validation_dir=self.dl_config.get("validation_directory"),
            seed=self.seed,
            valid_batch_size=25,
            nb_time_steps=self.nb_time_steps
        )

    def get_mlp(self):
        return pdeqx.arch.MLP(
            num_spatial_dims=self.advection_config["nb_dims"],
            in_channels=1,
            out_channels=1,
            num_points=self.advection_config["nb_points"],
            width_size=self.dl_config.get("width_size", 64),
            depth=self.dl_config.get("depth", 3),
            boundary_mode="dirichlet",
            key=jax.random.PRNGKey(0)
        )

    def get_optimizer(self):
        return optax.adam(self.dl_config.get("lr", 3e-4))

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
        update_out = tutils.update_fn(
            self.model,
            self.optimizer,
            u_prev,
            u_next,
            self.opt_state,
        )
        self.model, self.opt_state, batch_loss, loss_per_sample, grads = update_out
        for key, val in tutils.get_grads_stats(grads).items():
            self.tb_logger.log_scalar(f"Gradients/{key}", val, batch_id)
        logger.info(f"BATCH={batch_id} loss={batch_loss:.2e}")
        self.tb_logger.log_scalar("Loss/train", batch_loss.item(), batch_id)

        if self.sampler_type == "breed":
            loss_per_sample = torch.tensor(loss_per_sample.tolist())
            batch_loss = torch.tensor(batch_loss.item())
            batch_loss_relative = active_sampling.get_relative_loss(loss_per_sample)
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
            val_loss = 0.0
            count = 0
            for v_batch_id, valid_batch_data in enumerate(self.valid_dataloader):
                u_prev, u_next, sim_ids = valid_batch_data
                u_prev = jnp.asarray(u_prev)
                u_next = jnp.asarray(u_next)
                batch_loss, _ = tutils.loss_fn(self.model, u_prev, u_next) 
        
                val_loss += batch_loss.item()
                count += 1
            # endfor
            avg_val_loss = val_loss / count if count > 0 else 0.0
            self.tb_logger.log_scalar("Loss/valid", avg_val_loss, batch_id)
 
    @override
    def prepare_training_attributes(self):

        model = self.get_mlp()
        optimizer = self.get_optimizer()
        logger.info(f"Model parameters count: {pdeqx.count_parameters(model)}")

        return model, optimizer

    @override
    def process_simulation_data(self, msg, config_dict):
        u_prev = msg.data["preposition"]
        u_next = msg.data["position"]

        u_prev = u_prev.reshape(1, *self.mesh_shape)
        u_next = u_next.reshape(1, *self.mesh_shape)
        
        u_prev = np.array(u_prev, copy=True)
        u_next = np.array(u_next, copy=True)
        
        return u_prev, u_next, msg.simulation_id, msg.time_step
