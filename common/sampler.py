import os
import jax.random as jr
import numpy as np
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


class JaxSamplerMixIn:
    """Use this mix-in when you have your specific (non-APEBench) IC sampler.
    For example, the sine wave IC sampler is not used in APEBench."""
    def __init__(self, ic_config, is_valid=False):
        self.main_key = jr.PRNGKey(self.seed)
        self.make_keys()
        self.ic_config = ic_config
        self.is_valid = is_valid
        if self.is_valid:
            os.makedirs(VALIDATION_DIR, exist_ok=True)

    def make_keys(self):
        self.main_key, self.param_key = jr.split(self.main_key)

    # since jax relies on keys we need to change them while resampling,
    # otherwise it will produce the same values
    # note: This is only for required for DefaultBreeder class
    # as we don't sample more than once in a regular case.
    @override
    def sample(self, nb_samples=1):
        samples = super().sample(nb_samples)
        self.make_keys()

        return samples

    @abstractmethod
    def get_placeholders(self):
        raise NotImplementedError

    def make_sampled_ic_config(self, sampled_params):
        sampled_ic_config = self.ic_config
        placeholders = self.get_placeholders()
        assert len(sampled_params) == len(placeholders)
        # for example, ic_config = "sine;<amp>;<phs>;2.0;true;true"
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
        if self.is_valid and self.current_index == self.nb_sims:
            jnp.save(
                VALIDATION_INPUT_PARAM_FILE,
                self.parameters
            )
        return [
            f'--ic-config="{sampled_ic_config}"'
        ]


class SineWaveSamplerMixIn(JaxSamplerMixIn):

    def __init__(self, ic_config, is_valid=False):
        JaxSamplerMixIn.__init__(self, ic_config, is_valid)

    def get_placeholders(self):
        out = []
        for i in range(1, self.nb_params // 2 + 1):
            out.extend([f"<amp{i}>", f"<phs{i}>"])
        return out

    @override
    def sample(self, nb_samples=1):
        sampled_params = np.asarray(
            jr.uniform(
                self.param_key,
                shape=(nb_samples, self.nb_params),
                minval=self.l_bounds,
                maxval=self.u_bounds
            )
        )
        self.make_keys()
        return sampled_params
        

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


# class JaxAPEBenchSamplerFileSaver:
#     """Use this class when you want to use an IC set of
#     a specific APEBench scenario"""
#     def __init__(self, scenario):
#         self.scenario = scenario
# 
#     def produce_ic(self, nb_samples, is_valid=False):
#         if is_valid:
#             self.scenario.num_test_samples = nb_samples
#             return self.scenario.get_test_ic_set()
#         self.scenario.num_train_samples = nb_samples
#         return self.scenario.get_train_ic_set()
# 
#     def save_all(self, samples, is_valid=False):
#         if is_valid:
#             jnp.save(VALIDATION_INPUT_PARAM_FILE, samples)
#         else:
#             os.makedirs(IC_DIR, exist_ok=True)
#             for sim_id in samples:
#                 jnp.save(
#                     f"{IC_DIR}/sim{sim_id}.npy", samples[sim_id]
#                 )
# 


# class ScenarioClassicSampler(JaxAPEBenchSamplerFileSaver, StaticExperiment):
#     def __init__(self, scenario, is_valid=False, **kwargs):
#         StaticExperiment.__init__(self, **kwargs)
#         JaxAPEBenchSamplerFileSaver.__init__(self, scenario)
#         samples = self.produce_ic(self.nb_sims, is_valid)
#         self.set_parameters(np.asarray(samples))
# 
#     @override
#     def draw(self):
#         params = super().draw()
#         sim_id = self.current_index
#         ic_path = f"{IC_DIR}/sim{sim_id}.npy"
#         jnp.save(ic_path, params)
#         return [
#             f"--ic-path={ic_path}"
#         ]
# 
# 
# class ScenarioBreedSampler(JaxAPEBenchSamplerFileSaver, DefaultBreeder):
#     @override
#     def sample(self, nb_samples):
#         samples = self.produce_ic(nb_samples)
#         return samples
# 
#     @override
#     def draw(self):
#         params = super().draw()
#         sim_id = self.current_index
#         ic_path = f"{IC_DIR}/sim{sim_id}.npy"
#         jnp.save(ic_path, params)
#         return [
#             f"--ic-path={ic_path}"
#         ]
# 
