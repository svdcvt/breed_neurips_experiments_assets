import logging

from typing_extensions import override
import torch
import numpy as np
import jax
import jax.numpy as jnp
import optax
import pdequinox as pdeqx

from melissa.server.deep_learning.tensorboard_logger import (  # type: ignore
    TorchTensorboardLogger
)
from melissa.server.deep_learning.active_sampling.active_sampling_server \
    import ExperimentalDeepMelissaActiveSamplingServer

import train_utils
from custom_sampler import CustomICBreeder, CustomICUniformSampler


logger = logging.getLogger("melissa")


class BaseAPEBenchServer(ExperimentalDeepMelissaActiveSamplingServer):

    def __init__(self, config_dict):
        super().__init__(config_dict)

        # initialize tensorboardLogger with torch
        self._tb_logger = TorchTensorboardLogger(
            self.rank,
            disable=not self.dl_config["tensorboard"],
            debug=self.verbose_level >= 3
        )

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

    @override
    def checkpoint(self):
        pass

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

        return u_prev, u_next

    @override
    def train(self):
        dataloader = torch.utils.data.DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            drop_last=True,
            num_workers=0
        )

        opt_state = None
        for batch_id, batch in enumerate(dataloader):
            u_prev, u_next = batch
            u_prev = jnp.asarray(u_prev)
            u_next = jnp.asarray(u_next)
            self.model, opt_state, batch_loss = train_utils.update_fn(
                self.model,
                self.optimizer,
                u_prev,
                u_next,
                opt_state,
            )
            logger.info(f"BATCH={batch_id} loss={batch_loss}")
            scalar_loss = batch_loss.tolist()
            self.tb_logger.log_scalar("Loss/train", scalar_loss, batch_id)

    @override
    def _load_model_from_checkpoint(self):
        pass

    @override
    def _setup_environment_slurm(self):
        pass
