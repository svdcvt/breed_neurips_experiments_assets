from typing import Dict, Any

from melissa.server.sensitivity_analysis import (  # type: ignore
    SensitivityAnalysisServer
)


class OfflineServer(SensitivityAnalysisServer):
    """Use-case specific server"""

    def __init__(self, config_dict: Dict[str, Any]):
        super().__init__(config_dict)
        self.make_study_offline()
