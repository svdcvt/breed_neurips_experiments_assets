import apebench
import exponax as ex

IC_DICT = apebench.components.ic_dict
CUSTOM_IC_GENERATORS = {
    "sine": lambda ic_config, num_spatial_dims: ex.ic.SineWaves1d(
        num_spatial_dims=num_spatial_dims,
        amplitudes=(float(ic_config.split(";")[1]),),
        phases=(float(ic_config.split(";")[2]),),
        zero_mean=ic_config.split(";")[3].lower() == "true",
        max_one=ic_config.split(";")[4].lower() == "true",
    ),
}

IC_DICT.update(CUSTOM_IC_GENERATORS)


def make_ic(num_spatial_dims, num_channels, sampled_ic_config):

    """Modifying `apebench._base_scenario.BaseScenario.get_ic_generator()`
    to accept our sampled parameters."""

    def _get_single_channel(config):
        ic_name = config.split(";")[0]
        ic_gen = IC_DICT[ic_name.lower()](config, num_spatial_dims)
        return ic_gen

    ic_args = sampled_ic_config.split(";")
    if ic_args[0].lower() == "clamp":
        lower_bound = float(ic_args[1])
        upper_bound = float(ic_args[2])

        ic_gen = _get_single_channel(";".join(ic_args[3:]))
        ic_gen = ex.ic.ClampingICGenerator(
            ic_gen,
            limits=(lower_bound, upper_bound),
        )
    else:
        ic_gen = _get_single_channel(sampled_ic_config)

    multi_channel_ic_gen = ex.ic.RandomMultiChannelICGenerator(
        [
            ic_gen,
        ]
        * num_channels
    )

    return multi_channel_ic_gen
