# flake8: noqa
from typing import Dict, Any
from typing_extensions import override
import numpy as np
import logging

from melissa.server.sensitivity_analysis import SensitivityAnalysisServer  # type: ignore

from common import StaticHaltonSampler

logger = logging.getLogger("melissa")


class ValidationSampler(StaticHaltonSampler):
    @override
    def draw(self):
        amp, phs = super().draw()

        if self.current_index == self.nb_sims:
            np.save(
                "trajectories/input_parameters.npy",
                self.parameters
            )

        return [
            f"--amplitude={amp}",
            f"--phase={phs}",
        ]


class OfflineServer(SensitivityAnalysisServer):
    """Use-case specific server"""

    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__(config_dict)
        self.make_study_offline()
        study_options = config_dict["study_options"]
        self.seed = study_options["seed"]

        # amplitude, phase
        l_bounds = [-1.0, 0.0]
        u_bounds = [1.0, 2 * np.pi]

        self.set_parameter_sampler(
            sampler_t=ValidationSampler,
            seed=self.seed,
            l_bounds=l_bounds,
            u_bounds=u_bounds,
            dtype=np.float32,
        )
 
