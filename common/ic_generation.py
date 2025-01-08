import os
import jax.numpy as jnp
import exponax as ex
from abc import ABC, abstractmethod

from common import IC_DIR


class BaseICMaker(ABC):
    def __init__(self, config):
        self.config = config
        self.ic_maker = None

    def input_from_file(self):
        sim_id = os.environ["MELISSA_SIM_ID"]
        return jnp.load(f"{IC_DIR}/sim{sim_id}.npy")

    @abstractmethod
    def process_input(self, **input_fn_args):
        pass

    @abstractmethod
    def __call__(self, **input_fn_args):
        raise NotImplementedError


class SineWave(BaseICMaker):
    def __init__(self, config):
        super().__init__(config)
        config_parts = config.split(";")
        self.ic_maker = ex.ic.SineWaves1d(
            domain_extent=float(config_parts[3]),
            amplitudes=(float(config_parts[4]),),
            phases=(float(config_parts[5]),),
            zero_mean=config_parts[6].lower() == "true",
            max_one=config_parts[7].lower() == "true",
        )

    def process_input(self,
                      num_spatial_dims,
                      domain_extent,
                      num_points,
                      **extra_args):

        return ex.make_grid(
            num_spatial_dims,
            domain_extent,
            num_points,
            **extra_args
        )

    def __call__(self,
                 num_spatial_dims,
                 domain_extent,
                 num_points,
                 **extra_args):

        input_ = self.process_input(
            num_spatial_dims,
            domain_extent,
            num_points,
            **extra_args
        )

        return self.ic_maker(input_)


CUSTOM_IC_MAKERS = {
    "sine": SineWave,
    "fourier": "save ex.ic.RandomTruncatedFourierSeries to files",
    "diffused": "save ex.ic.DiffusedNoise to files",
    "grf": "save ex.ic.GaussianRandomField to files",
}


def make_ic(sampled_ic_config, **input_fn_args):
    ic_type = sampled_ic_config.split(";")[0]
    ic_maker = CUSTOM_IC_MAKERS[ic_type]
    if isinstance(ic_maker, str):
        print(ic_maker)
        os.exit(1)
    return ic_maker(**input_fn_args)
