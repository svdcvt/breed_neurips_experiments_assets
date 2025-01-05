import logging

from typing_extensions import override
import torch

from melissa.server.deep_learning.tensorboard_logger import (  # type: ignore
    TorchTensorboardLogger
)
from melissa.server.deep_learning import active_sampling
from melissa.server.deep_learning.active_sampling.active_sampling_server \
    import ExperimentalDeepMelissaActiveSamplingServer


logger = logging.getLogger("melissa")


class BaseAPEBenchServer(ExperimentalDeepMelissaActiveSamplingServer):

    def __init__(self, config_dict):
        super().__init__(config_dict)

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

    @override
    def checkpoint(self):
        pass

    @override
    def _load_model_from_checkpoint(self):
        pass

    @override
    def _setup_environment_slurm(self):
        pass

