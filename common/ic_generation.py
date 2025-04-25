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
    def __call__(self, **input_fn_args):
        raise NotImplementedError


class SineWave(BaseICMaker):
    def __init__(self,
                 domain_extent,
                 num_points,
                 sampled_ic_config):

        super().__init__(
            1,
            domain_extent,
            num_points,
            sampled_ic_config
        )
        # ic_config "sine;<amp>;<phs>;true;true"
        config_parts = self.sampled_ic_config.split(";")
        self.ic_maker = ex.ic.SineWaves1d(
            domain_extent=self.domain_extent,
            amplitudes=(float(config_parts[1]),),
            wavenumbers=(1,),
            phases=(float(config_parts[2]),),
            std_one=config_parts[-2].lower() == "true",
            max_one=False #config_parts[-1].lower() == "true",
        )

    def __call__(self, **extra_args):
        grid = ex.make_grid(
            self.num_spatial_dims,
            self.domain_extent,
            self.num_points,
            **extra_args
        )
        return self.ic_maker(grid)


class SupSineWave(SineWave):
    def __init__(self,
                 domain_extent,
                 num_points,
                 sampled_ic_config):

        super().__init__(
            domain_extent,
            num_points,
            sampled_ic_config
        )
        # ic_config "sine_sup;<amp1>;<phs1>;<amp2>;<phs2>;....;<ampk>;<phsk>;true;true"
        config_parts = self.sampled_ic_config.split(";")
        amplitudes = tuple([float(amp) for amp in config_parts[1:-2:2]])
        phases = tuple([float(phs) for phs in config_parts[2:-2:2]])
        wavenumbers = tuple(range(1, len(amplitudes) + 1))
        self.ic_maker = ex.ic.SineWaves1d(
            domain_extent=self.domain_extent,
            amplitudes=amplitudes,
            wavenumbers=wavenumbers,
            phases=phases,
            std_one=config_parts[-2].lower() == "true",
            max_one=False #config_parts[-1].lower() == "true",
        )
        # !!! INFO we assume that amplitudes are abs-maximum 1 !!!
        is_max_one = config_parts[-1].lower() == "true"
        if is_max_one:
            print(
                "WARNING: amplitudes are assumed to be abs-maximum 1, "
                "normalisation is applied as IC / number of parameters"
            )
        self.normalise = len(amplitudes) if is_max_one else None

    def __call__(self, **extra_args):
        grid = ex.make_grid(
            self.num_spatial_dims,
            self.domain_extent,
            self.num_points,
            **extra_args
        )
        ic_mesh = self.ic_maker(grid)
        if self.normalise is not None:
            ic_mesh = ic_mesh / self.normalise
        return ic_mesh

class SineCosWaves2D(BaseICMaker):
    def __init__(self,
                 domain_extent,
                 num_points,
                 sampled_ic_config):

        super().__init__(
            2,
            domain_extent,
            num_points,
            sampled_ic_config
        )
        # ic_config "sine_cos_2d;<amp>;<phs>;
        config_parts = self.sampled_ic_config.split(";")
        amp = float(config_parts[1])
        phs = float(config_parts[2])
        self.ic_maker = lambda grid: (
            amp
            * jnp.sin(2 * 2 * jnp.pi * grid[0:1] / self.domain_extent + phs)
            * jnp.cos(3 * 2 * jnp.pi * grid[1:2] / self.domain_extent + phs)
        )

    def __call__(self, **extra_args):
        grid = ex.make_grid(
            self.num_spatial_dims,
            self.domain_extent,
            self.num_points,
            **extra_args
        )
        return self.ic_maker(grid)


CUSTOM_IC_MAKERS = {
    "sine": SineWave,
    "sine_sup": SupSineWave,
    "sine_cos_2d": SineCosWaves2D
}


def get_ic_maker(domain_extent,
                 num_points,
                 sampled_ic_config,
                 **extra_args):

    ic_type = sampled_ic_config.split(";")[0]
    ic_maker = CUSTOM_IC_MAKERS[ic_type]

    if isinstance(ic_maker, str):
        print(ic_maker)
        os.exit(1)

    return ic_maker(
        domain_extent,
        num_points,
        sampled_ic_config,
        **extra_args
    )
