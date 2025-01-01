import os
import logging

from typing_extensions import override
import numpy as np
import jax

from melissa.server.deep_learning.active_sampling.active_sampling_server import \
    ExperimentalDeepMelissaActiveSamplingServer
from custom_sampler import CustomICBreeder, CustomICUniformSampler


logger = logging.getLogger("melissa")


class BaseAPEBenchServer(ExperimentalDeepMelissaActiveSamplingServer):

    def __init__(self, config_dict):
        super().__init__(config_dict)

        study_options = config_dict["study_options"]
        self.seed = study_options["seed"]
        self.l_bounds = study_options["lower_bounds"]
        self.u_bounds = study_options["upper_bounds"]

        # self.jax_keys = jax.random.PRNGKey(self.seed)

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

    @override
    def checkpoint(self):
        pass

    @override
    def prepare_training_attributes(self):

        model = None
        optimizer = None
        return model, optimizer

    @override
    def process_simulation_data(self):
        pass

    @override
    def train(self):
        pass

    @override
    def _load_model_from_checkpoint(self):
        pass

    @override
    def _setup_environment_slurm(self):
        pass
