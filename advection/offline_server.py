import numpy as np
from typing import Dict, Any

from melissa.server.sensitivity_analysis import (  # type: ignore
    SensitivityAnalysisServer
)


class OfflineServer(SensitivityAnalysisServer):
    """Use-case specific server"""

    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__(config_dict)
        self.make_study_offline()

        self.set_parameter_sampler(
            sampler_t=self.sampler_t,  # set this in child
            is_valid=True,
            **self.breed_params,
            seed=self.seed,
            l_bounds=self.l_bounds,
            u_bounds=self.u_bounds,
            dtype=np.float32
        )
