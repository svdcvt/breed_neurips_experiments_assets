import numpy as np
import jax
import jax.numpy as jnp
from typing_extensions import override

from melissa.server.parameters import (  # type: ignore
        StaticExperiment,
)
from melissa.server.deep_learning.active_sampling import (  # type: ignore
    DefaultBreeder
)


class ICSamplerMixIn:

    def __init__(self, **kwargs):
        self.amp_key, self.phs_key = self.make_keys(kwargs["seed"])
        super().__init__(**kwargs)

    def make_keys(self, seed):
        main_key = jax.random.PRNGKey(seed)
        return jax.random.split(main_key, 2)

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

    @override
    def draw(self):
        amp, phs = super().draw()
        return [
            f"--amplitude={amp}",
            f"--phase={phs}",
        ]


# for dynamic parameter sampler draw() -> sample(1) is called which produces
# the same parameters in case of `jax.random.uniform` due to having the same key
# instead, we sample them in bulk with `StaticExperiment` which calls `sample(self.nb_sims)`
class CustomICUniformSampler(ICSamplerMixIn, StaticExperiment):
    pass


class CustomICBreeder(ICSamplerMixIn, DefaultBreeder):

    # since jax relies on keys we need to change them while resampling,
    # otherwise it will produce the same values
    def get_non_breed_samples(self, nb_samples):
        samples = super().get_non_breed_samples(nb_samples)
        self.seed += 1
        self.amp_key, self.phs_key = self.make_keys(self.seed)

        return samples

