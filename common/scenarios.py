import jax
import exponax  # noqa
import apebench
import ic_generation as icgen


def get_exponax_stepper(scenario, **stepper_config):
    stepper_t = eval(f"exponax.stepper.{scenario.__class__.__name__}")
    return stepper_t(**stepper_config)


def get_apebench_scenario(name, **scenario_config):
    return apebench.scenarios.scenario_dict[name](**scenario_config)


class MelissaSpecificScenario:
    def __init__(self,
                 scenario_name,
                 use_exponax_stepper=False,
                 sampled_ic_config=None,
                 network_config="MLP;64;3;relu",
                 stepper_config={},
                 **scenario_config):

        self.scenario_name = scenario_name
        self.scenario = get_apebench_scenario(
            self.scenario_name,
            **scenario_config
        )

        if use_exponax_stepper:
            self.stepper = get_exponax_stepper(self.scenario, **stepper_config)
        else:
            self.stepper = self.scenario.get_ref_stepper()

        self.network_config = network_config
        self.num_spatial_dims = self.scenario.num_spatial_dims

        # modify these in stepper_config = {}
        self.domain_extent = self.stepper.domain_extent
        self.dt = self.stepper.dt

        self.train_temporal_horizon = self.scenario.train_temporal_horizon
        self.num_channels = self.scenario.num_channels
        self.num_points = self.scenario.num_points
        self.sampled_ic_config = sampled_ic_config \
            if sampled_ic_config else scenario_config["ic_config"]

    def get_shape(self):
        return (self.num_channels,) + \
            (self.scenario.num_points,) * self.num_spatial_dims

    def get_network(self):
        return self.scenario.get_network(
            network_config=self.network_config,
            key=jax.random.PRNGKey(0)
        )

    def get_optimizer(self):
        return self.scenario.get_optimizer()

    def get_stepper(self):
        return self.stepper

    def get_ic_mesh(self, **input_fn_config):

        ic_maker = icgen.get_ic_maker(
            self.domain_extent,
            self.num_points,
            self.sampled_ic_config,
        )
        assert ic_maker.num_spatial_dims == self.num_spatial_dims, \
            f"This is due to having IC with {ic_maker.num_spatial_dims}D " \
            f"while the scenario is set with {self.num_spatial_dims}D" \
            "Adjust this in `['scenario_config']['stepper_config']` option."

        return ic_maker(**input_fn_config)
