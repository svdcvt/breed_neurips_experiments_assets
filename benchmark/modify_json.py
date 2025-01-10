import os
import rapidjson as json
import argparse


def update_breed_params(config, period, sigma, start, end, breakpoint, window_size):
    config['active_sampling_config']['nn_updates'] = int(period)
    config['active_sampling_config']['breed_params']['sigma'] = sigma
    config['active_sampling_config']['breed_params']['start'] = start
    config['active_sampling_config']['breed_params']['end'] = end
    config['active_sampling_config']['breed_params']['breakpoint'] = int(breakpoint)
    config['active_sampling_config']['breed_params']['sliding_window_size'] = int(window_size)


def modify_json(input_config_file,
                output_config_file,
                output_dir,
                scenario_config,
                sampler=None,
                validation_directory=None,
                # sigma,
                # start,
                # end,
                # breakpoint,
                # period,
                # window_size,
                offline=False,
                seed=None):

    CONFIG_PARSE_MODE = json.PM_COMMENTS | json.PM_TRAILING_COMMAS
    with open(input_config_file, 'r') as file:
        config = json.load(file, parse_mode=CONFIG_PARSE_MODE)

    config['output_dir'] = output_dir
    if seed is not None:
        config['study_options']['seed'] = seed

    config['study_options']["scenario_config"].update(scenario_config)

    if not offline:
        if validation_directory is not None:
            config["dl_config"]["validation_directory"] = validation_directory
        config['sampler_type'] = sampler
        if sampler == "breed":
            pass
            # update_bree_params(
            #     config, period, sigma, start, end, breakpoint, window_size
            # )
        else:
            config['active_sampling_config']['nn_updates'] = -1

    fname = os.path.split(output_config_file)[-1]
    config['client_config']['preprocessing_commands'].append(
        f"export CONFIG_FILE={fname}"
    )
    with open(output_config_file, 'w') as file:
        json.dump(config, file, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify JSON configuration file")
    parser.add_argument('input_config_file', type=str, help='Path to the JSON configuration file')
    parser.add_argument('output_config_file', type=str, help='Path to the JSON configuration file')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--sampler', type=str, default=None, help='Sampler type')
    # parser.add_argument('--sigma', type=float, default=None, help='Sigma value')
    # parser.add_argument('--start', type=float, default=None, help='Start value')
    # parser.add_argument('--end', type=float, default=None, help='End value')
    # parser.add_argument('--breakpoint', type=int, default=None, help='Breakpoint value')
    # parser.add_argument('--period', type=float, default=None, help='Active sampling period value')
    # parser.add_argument('--window-size', type=int, default=None, help='Sliding window size value')
    parser.add_argument('--seed', type=int, default=None, help='Random seed value')
    parser.add_argument('--offline', action='store_true', help='Set this when you want to store the data.')

    args = parser.parse_args()

    modify_json(**vars(args))
