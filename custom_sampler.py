from typing_extensions import override

from melissa.server.parameters import RandomUniform
from melissa.server.deep_learning.active_sampling import DefaultBreeder


class ICSamplerMixIn:

    @override
    def draw(self):
        return [
            f"--velocity={super().draw()[0]}"
        ]


class CustomICUniformSampler(ICSamplerMixIn, RandomUniform):
    pass


class CustomICBreeder(ICSamplerMixIn, DefaultBreeder):
    pass
