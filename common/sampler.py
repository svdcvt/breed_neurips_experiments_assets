import jax
import numpy as np
import jax.numpy as jnp
from abc import abstractmethod
from typing_extensions import override

from melissa.server.parameters import (  # type: ignore
    BaseExperiment,
)
from melissa.server.deep_learning.active_sampling import (  # type: ignore
    DefaultBreeder
)


def replace_with_error_handling(original_str, old, new):
    if old not in original_str:
        raise ValueError(f"The substring '{old}' was not found in the string.")
    return original_str.replace(old, new)


class JaxAPEBenchSamplerMixIn:

    def __init__(self, ic_config):
        self.param_keys = self.make_keys()
        self.ic_config = ic_config

    def make_keys(self):
        main_key = jax.random.PRNGKey(self.seed)
        return jax.random.split(main_key, self.nb_parameters)

    # since jax relies on keys we need to change them while resampling,
    # otherwise it will produce the same values
    # Note: This is only for DefaultBreeder class
    @override
    def get_non_breed_samples(self, nb_samples):
        samples = super().get_non_breed_samples(nb_samples)
        self.seed += 1
        self.param_keys = self.make_keys()

        return samples

    @abstractmethod
    def get_placeholders(self):
        raise NotImplementedError

    def make_sampled_ic_config(self, sampled_params):
        sampled_ic_config = self.ic_config
        placeholders = self.get_placeholders()
        assert len(sampled_params) == len(placeholders)
        # for example, ic_config = "sine;<amp>;<phs>;true;true"
        # will replace the placeholders with actual sampled parameters.
        for plhd, param in zip(placeholders, sampled_params):
            sampled_ic_config = replace_with_error_handling(
                sampled_ic_config,
                plhd,
                str(param)
            )
        return sampled_ic_config

    @override
    def draw(self):
        sampled_params = super().draw()
        sampled_ic_config = self.make_sampled_ic_config(sampled_params)
        return [
            f"--ic-config={sampled_ic_config}"
        ]


class SineWaveSamplerMixIn(JaxAPEBenchSamplerMixIn):

    def __init__(self, ic_config):
        JaxAPEBenchSamplerMixIn.__init__(self, ic_config)
        self.amp_key, self.phs_key = self.param_keys

    def get_placeholders(self):
        return ["<amp>", "<phs>"]

    @override
    def sample(self, nb_samples=1):
        amp = jax.random.uniform(
            self.amp_key, (nb_samples,),
            minval=self.l_bounds[0], maxval=self.u_bounds[0]
        )
        phs = jax.random.uniform(
            self.phs_key, (nb_samples,),
            minval=self.l_bounds[1], maxval=self.u_bounds[1]
        )

        return np.asarray(
            jnp.stack((amp, phs), axis=1).squeeze()
        ).astype(self.dtype)


class SineWaveClassicSampler(SineWaveSamplerMixIn, BaseExperiment):
    def __init__(self, ic_config, **kwargs):
        BaseExperiment.__init__(self, **kwargs)
        SineWaveSamplerMixIn.__init__(self, ic_config)


class SineWaveBreedSampler(SineWaveSamplerMixIn, DefaultBreeder):
    def __init__(self, ic_config, **kwargs):
        DefaultBreeder.__init__(self, **kwargs)
        SineWaveSamplerMixIn.__init__(self, ic_config)


CLASSIC_SAMPLERS = {
    "sine": SineWaveClassicSampler
}

BREED_SAMPLERS = {
    "sine": SineWaveBreedSampler
}


def get_sampler_class_type(ic_type: str, is_breed: bool):
    d = BREED_SAMPLERS if is_breed else CLASSIC_SAMPLERS
    return d[ic_type]
