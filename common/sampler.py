from melissa.server.parameters import (  # type: ignore
    StaticExperiment,
    HaltonSamplerMixIn
)
from melissa.server.deep_learning.active_sampling import (  # type: ignore
    DefaultBreeder
)


class JaxSpecificBreeder(DefaultBreeder):

    # since jax relies on keys we need to change them while resampling,
    # otherwise it will produce the same values
    def get_non_breed_samples(self, nb_samples):
        samples = super().get_non_breed_samples(nb_samples)
        self.seed += 1
        self.amp_key, self.phs_key = self.make_keys(self.seed)

        return samples


class StaticHaltonSampler(HaltonSamplerMixIn, StaticExperiment):
    """Convenience class to store all generated parameters."""
    # since we inherit SA server, we remove the parameters for pick freeze
    # they are by-default passed in the SensitivityAnalysisServer class.
    def __init__(self, **kwargs):
        del kwargs["apply_pick_freeze"], kwargs["second_order"]
        StaticExperiment.__init__(self, **kwargs)
        HaltonSamplerMixIn.__init__(self)
