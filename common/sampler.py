import os
import jax.random as jr
import numpy as np
import jax.numpy as jnp
from abc import abstractmethod
from typing_extensions import override

from melissa.server.parameters import (  # type: ignore
    BaseExperiment,
    StaticExperiment
)
from melissa.server.deep_learning.active_sampling import (  # type: ignore
    DefaultBreeder
)

from common import VALDIATION_INPUT_PARAM_FILE, IC_DIR


def replace_with_error_handling(original_str, old, new):
    if old not in original_str:
        raise ValueError(f"The substring '{old}' was not found in the string.")
    return original_str.replace(old, new)


class JaxAPEBenchSamplerMixIn:

    def __init__(self, ic_config):
        self.param_keys = self.make_keys()
        self.ic_config = ic_config

    def make_keys(self):
        main_key = jr.PRNGKey(self.seed)
        return jr.split(main_key, self.nb_parameters)

    # since jax relies on keys we need to change them while resampling,
    # otherwise it will produce the same values
    # Note: This is only for DefaultBreeder class
    @override
    def get_non_breed_samples(self, nb_samples):
        samples = super().get_non_breed_samples(nb_samples)
        self.seed = np.random.randint(
            low=0,
            high=10000,
            size=1
        ).item()
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


class JaxAPEBenchSamplerFileSaver:
    def __init__(self, scenario):
        self.scenario = scenario

    def produce_ic(self, nb_samples, is_valid=False):
        if is_valid:
            self.scenario.num_test_samples = nb_samples
            return self.scenario.get_test_ic_set()
        self.scenario.num_train_samples = nb_samples
        return self.scenario.get_train_ic_set()

    def save_all(self, samples, is_valid=False):
        if is_valid:
            jnp.save(VALDIATION_INPUT_PARAM_FILE, samples)
        else:
            os.makedirs(IC_DIR, exist_ok=True)
            for sim_id in samples:
                jnp.save(
                    f"{IC_DIR}/sim{sim_id}.npy", samples[sim_id]
                )


class SineWaveSamplerMixIn(JaxAPEBenchSamplerMixIn):

    def __init__(self, ic_config):
        JaxAPEBenchSamplerMixIn.__init__(self, ic_config)
        self.amp_key, self.phs_key = self.param_keys

    def get_placeholders(self):
        return ["<amp>", "<phs>"]

    @override
    def sample(self, nb_samples=1):
        amp = jr.uniform(
            self.amp_key, (nb_samples,),
            minval=self.l_bounds[0], maxval=self.u_bounds[0]
        )
        phs = jr.uniform(
            self.phs_key, (nb_samples,),
            minval=self.l_bounds[1], maxval=self.u_bounds[1]
        )

        return np.asarray(
            jnp.stack((amp, phs), axis=1).squeeze()
        ).astype(self.dtype)


class ScenarioClassicSampler(JaxAPEBenchSamplerFileSaver, StaticExperiment):
    def __init__(self, scenario, is_valid=False, **kwargs):
        StaticExperiment.__init__(self, **kwargs)
        JaxAPEBenchSamplerFileSaver.__init__(self, scenario)
        samples = self.produce_ic(self.nb_sims, is_valid)
        self.set_parameters(np.asarray(samples))

    @override
    def draw(self):
        params = super().draw()
        sim_id = self.current_index
        ic_path = f"{IC_DIR}/sim{sim_id}.npy"
        jnp.save(ic_path, params)
        return [
            f"--ic-path={ic_path}"
        ]


class ScenarioBreedSampler(JaxAPEBenchSamplerFileSaver, DefaultBreeder):
    @override
    def get_non_breed_samples(self, nb_samples):
        samples = self.produce_ic(nb_samples)
        return samples

    @override
    def draw(self):
        params = super().draw()
        sim_id = self.current_index
        ic_path = f"{IC_DIR}/sim{sim_id}.npy"
        jnp.save(ic_path, params)
        return [
            f"--ic-path={ic_path}"
        ]


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
