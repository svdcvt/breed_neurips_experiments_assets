import ic_generator as icgen
from apebench.scenarios.difficulty import (
    Advection,
    Diffusion,
    AdvectionDiffusion,
)


SCENARIOS = {
    "advection": Advection,
    "diffusion": Diffusion,
    "advection_diffusion": AdvectionDiffusion
}


def get_scenario(name, **scenario_config):
    if name not in SCENARIOS:
        raise ValueError(
            f"Unknown scenario: {name}. "
            f"Available scenarios: {', '.join(SCENARIOS.keys())}"
        )
    return SCENARIOS[name](**scenario_config)


class MelissaSpecificScenario:
    def __init__(self, scenario_name, **scenario_config):
        self.scenario_name = scenario_name
        self.scenario = get_scenario(self.scenario_name, **scenario_config)
        self.num_spatial_dims = self.scenario.num_spatial_dims
        self.num_channels = self.scenario.num_channels

    def get_shape(self):
        return (self.num_channels,) + \
            (self.scenario.num_points,) * self.num_spatial_dims

    def get_network(self, network_config: str, key):
        return self.scenario.get_network(
            network_config=network_config,
            key=key
        )

    def get_optimizer(self):
        return self.scenario.get_optimizer()

    def get_stepper(self):
        return self.scenario.get_stepper()

    def get_ic(self, sampled_ic_config: str):
        return icgen.make_ic(
            self.num_spatial_dims,
            self.num_channels,
            sampled_ic_config
        )
