import logging

from typing_extensions import override
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pdequinox as pdeqx

import train_utils
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

        sampler_type = config_dict.get("sampler_type", "uniform")
        sampler_t = \
            CustomICBreeder if sampler_type == "breed" \
            else CustomICUniformSampler

        self.set_parameter_sampler(
            sampler_t=sampler_t,
            seed=self.seed,
            l_bounds=self.l_bounds,
            u_bounds=self.u_bounds,
            dtype=np.float32
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
        opt_state = None
        for batch_id, batch in enumerate(self.train_dataloader):
            u_prev, u_next, sim_id, t_step = batch
            u_prev = jnp.asarray(u_prev)
            u_next = jnp.asarray(u_next)
            self.model, opt_state, batch_loss = train_utils.update_fn(
                self.model,
                self.optimizer,
                u_prev,
                u_next,
                opt_state,
            )
            logger.info(f"BATCH={batch_id} loss={batch_loss:.3e}")
            scalar_loss = batch_loss.tolist()
            self.tb_logger.log_scalar("Loss/train", scalar_loss, batch_id)

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

        return u_prev, u_next, msg.simulation_id, msg.time_step
