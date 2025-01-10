import os
import jax.numpy as jnp
from abc import abstractmethod
from typing_extensions import override

from melissa.server.parameters import (  # type: ignore
    StaticExperiment
)
from melissa.server.deep_learning.active_sampling import (  # type: ignore
    DefaultBreeder
)

from constants import VALIDATION_DIR, VALIDATION_INPUT_PARAM_FILE


def replace_with_error_handling(original_str, old, new):
    if old not in original_str:
        raise ValueError(f"The substring '{old}' was not found in the string.")
    return original_str.replace(old, new)


class BaseCustomSamplerMixIn:
    """Base class for sampling with custom samplers using placeholder strings
    ic_config."""
    def __init__(self, ic_config, is_valid=False):
        self.ic_config = ic_config
        self.is_valid = is_valid
        if self.is_valid:
            os.makedirs(VALIDATION_DIR, exist_ok=True)

    @abstractmethod
    def get_placeholders(self):
        raise NotImplementedError

    def make_sampled_ic_config(self, sampled_params):
        placeholders = self.get_placeholders()
        assert len(sampled_params) == len(placeholders)
        # for example, ic_config = "sine;<amp>;<phs>;2.0;true;true"
        # will replace the placeholders with actual sampled parameters.
        sampled_ic_config = self.ic_config
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
        if self.is_valid and self.current_index == self.nb_sims:
            jnp.save(
                VALIDATION_INPUT_PARAM_FILE,
                self.parameters
            )
        return [
            f'--ic-config="{sampled_ic_config}"'
        ]


#################################################################
# when adding a new sampler, follow the same approach like below
#################################################################

class SineWaveSamplerMixIn(BaseCustomSamplerMixIn):

    def __init__(self, ic_config, is_valid=False, **kwargs):
        BaseCustomSamplerMixIn.__init__(self, ic_config, is_valid)

    def get_placeholders(self):
        out = []
        for i in range(1, self.nb_params // 2 + 1):
            out.extend([f"<amp{i}>", f"<phs{i}>"])
        return out


class SineWaveClassicSampler(SineWaveSamplerMixIn, StaticExperiment):
    def __init__(self, ic_config, is_valid=False, **kwargs):
        StaticExperiment.__init__(self, **kwargs)
        SineWaveSamplerMixIn.__init__(self, ic_config, is_valid)


class SineWaveBreedSampler(SineWaveSamplerMixIn, DefaultBreeder):
    def __init__(self, ic_config, is_valid=False, **kwargs):
        DefaultBreeder.__init__(self, **kwargs)
        SineWaveSamplerMixIn.__init__(self, ic_config)


CLASSIC_SAMPLERS = {
    "sine": SineWaveClassicSampler,
    "sine_sup": SineWaveClassicSampler
}

BREED_SAMPLERS = {
    "sine": SineWaveBreedSampler,
    "sine_sup": SineWaveBreedSampler
}


def get_sampler_class_type(ic_type: str, is_breed: bool):
    d = BREED_SAMPLERS if is_breed else CLASSIC_SAMPLERS
    return d[ic_type]
