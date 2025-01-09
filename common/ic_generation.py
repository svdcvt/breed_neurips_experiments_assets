import os
import jax.numpy as jnp
import exponax as ex
from abc import ABC, abstractmethod

from constants import IC_DIR


class BaseICMaker(ABC):
    def __init__(self,
                 num_spatial_dims,
                 domain_extent,
                 num_points,
                 sampled_ic_config):
        self.num_spatial_dims = num_spatial_dims
        self.domain_extent = domain_extent
        self.num_points = num_points
        self.sampled_ic_config = sampled_ic_config
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
    def __init__(self,
                 num_spatial_dims,
                 domain_extent,
                 num_points,
                 sampled_ic_config):

        super().__init__(num_spatial_dims, domain_extent,
                         num_points, sampled_ic_config)
        # ic_config "sine;<amp>;<phs>;true;true"
        config_parts = self.sampled_ic_config.split(";")
        self.ic_maker = ex.ic.SineWaves1d(
            domain_extent=self.domain_extent,
            amplitudes=(float(config_parts[1]),),
            wavenumbers=(1,),
            phases=(float(config_parts[2]),),
            std_one=config_parts[-2].lower() == "true",
            max_one=config_parts[-1].lower() == "true",
        )

    def process_input(self, **extra_args):

        return ex.make_grid(
            self.num_spatial_dims,
            self.domain_extent,
            self.num_points,
            **extra_args
        )

    def __call__(self, **extra_args):

        input_ = self.process_input(**extra_args)
        return self.ic_maker(input_)


class SupSineWave(SineWave):
    def __init__(self,
                 num_spatial_dims,
                 domain_extent,
                 num_points,
                 sampled_ic_config):

        super().__init__(num_spatial_dims, domain_extent,
                         num_points, sampled_ic_config)
        # ic_config "sine;<amp>;<phs>;true;true"
        config_parts = self.sampled_ic_config.split(";")
        self.ic_maker = ex.ic.SineWaves1d(
            domain_extent=self.domain_extent,
            amplitudes=(float(config_parts[1]), float(config_parts[2])),
            wavenumbers=(1, 2),
            phases=(float(config_parts[3]), float(config_parts[4])),
            std_one=config_parts[-2].lower() == "true",
            max_one=config_parts[-1].lower() == "true",
        )


CUSTOM_IC_MAKERS = {
    "sine": SineWave,
    "sine_sup": SupSineWave
}


def get_ic_maker(num_spatial_dims,
            domain_extent,
            num_points,
            sampled_ic_config,
            **extra_args):
    ic_type = sampled_ic_config.split(";")[0]
    ic_maker = CUSTOM_IC_MAKERS[ic_type]
    if isinstance(ic_maker, str):
        print(ic_maker)
        os.exit(1)
    return ic_maker(
        num_spatial_dims,
        domain_extent,
        num_points,
        sampled_ic_config,
        **extra_args
    )
