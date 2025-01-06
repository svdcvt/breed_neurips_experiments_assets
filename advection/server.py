import logging

import numpy as np
import jax
import optax
import pdequinox as pdeqx

from sampler import CustomICBreeder, CustomICUniformSampler
from common import BaseAPEBenchServer


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
