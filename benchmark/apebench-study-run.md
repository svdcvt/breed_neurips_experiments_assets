## Create configurations
Ensure that you modify your `$ROOT/default_*.json` files before generating configuration for APEBench benchmark run set in `apebench_study.py`.

_Note: We rename `net -> network_config` and `scenario -> scenario_config` adapting to melissa configuration keys._


### Create
One can override default root directories and default input files from command line arguments of the script.

Create the validation files per stepper type, gamma value, etc.,
```bash
python3 generate_configs.py --offline
```
```bash
configs/validation
└── diff_adv
    ├── gamma_0_5
    │   └── config_diff_adv_0_5.json
    ├── gamma_10_5
    │   └── config_diff_adv_10_5.json
    └── gamma_2_5
        └── config_diff_adv_2_5.json

5 directories, 3 files
```

Create the training files per stepper type, network arch, gamma value, and the sampling type (breed, uniform), etc.
```bash
python3 generate_configs.py
```
```bash
configs/train/
└── diff_adv
    ├── Conv_34_0_relu
    │   ├── gamma_0_5
    │   │   ├── config_diff_adv_Conv_34_0_relu_0_5_breed.json
    │   │   └── config_diff_adv_Conv_34_0_relu_0_5_uniform.json
    │   ├── gamma_10_5
    │   │   ├── config_diff_adv_Conv_34_0_relu_10_5_breed.json
    │   │   └── config_diff_adv_Conv_34_0_relu_10_5_uniform.json
    │   └── gamma_2_5
    │       ├── config_diff_adv_Conv_34_0_relu_2_5_breed.json
    │       └── config_diff_adv_Conv_34_0_relu_2_5_uniform.json
# truncated
```

_Remember to set `CONFIG_FILE` environment variable on the client side as this helps the client to load scenario with the same configuration from CONFIG_FILE `json` file. Already done while generating json files in `modify_json.py`_

## Run all scripts under specified folder recursively
The script will traverse all folders recursively from the give path and looks for `*.json` that are not in the output folder of the previous runs. If the job is already submitted it will not be submitted again based on extracted `status 0` string from the `melissa_server_0.log`.
```bash
# ./by_folder.sh <folder-to-search> <submission-script>
./by_folder.sh configs/train/diff_adv/Conv_34_0_relu/gamma_10_5 ../jz_semig_job.sh
```

### Cleanup
Run this to clear unnecessary folders for all the `STUDY_OUT*/` under a specified directory. This retains the tensorboard logs and the pandas pickle stored in `STUDY_OUT*/tensorboard` folder.
```bash
./cleanup.sh <folder-to-clean>
```

