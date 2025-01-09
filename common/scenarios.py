import jax
import apebench  # type: ignore
import exponax  # type: ignore

import ic_generation as icgen


def get_apebench_scenario(name, **scenario_config):
    try:
        scenario_t = eval(f"{name}")
        scenario = scenario_t(**scenario_config)
        return scenario
    except Exception as e:
        print(e)
        raise e


def get_exponax_stepper(name,
                        num_spatial_dims,
                        domain_extent,
                        num_points,
                        dt,
                        **stepper_config):
    try:
        stepper_t = eval(f"{name}")
        stepper = stepper_t(
            num_spatial_dims,
            domain_extent,
            num_points,
            dt,
            **stepper_config
        )
        return stepper
    except Exception as e:
        print(e)


class MelissaSpecificScenario:
    def __init__(self,
                 scenario_name,
                 stepper_name=None,
                 sampled_ic_config=None,
                 domain_extent=None,
                 dt=None,
                 network_config="MLP;64;3;relu",
                 stepper_config={},
                 input_fn_config={},
                 **scenario_config):
        self.scenario_name = scenario_name
        self.stepper_name = stepper_name
        self.scenario = get_apebench_scenario(
            self.scenario_name,
            **scenario_config
        )
        self.network_config = network_config
        self.num_spatial_dims = self.scenario.num_spatial_dims
        self.domain_extent = domain_extent
        if (
            self.domain_extent is None
            and hasattr(self.scenario, "domain_extent")
        ):
            self.domain_extent = self.scenario.domain_extent
        self.num_channels = self.scenario.num_channels
        self.num_points = self.scenario.num_points
        self.dt = dt if dt else self.scenario.dt
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

    def get_stepper(self, **stepper_config):
        if self.stepper_name:
            return get_exponax_stepper(
                self.stepper_name,
                self.num_spatial_dims,
                self.domain_extent,
                self.num_points,
                self.dt,
                **stepper_config
            )
        return self.scenario.get_ref_stepper()

    def make_ic(self, **input_fn_config):
        return icgen.make_ic(
            self.num_spatial_dims,
            self.domain_extent,
            self.num_points,
            self.sampled_ic_config,
        )(**input_fn_config)
