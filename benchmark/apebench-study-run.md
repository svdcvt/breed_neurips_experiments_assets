## Create configurations
Ensure that you modify your `$ROOT/default_*.json` files before generating configuration for APEBench benchmark run set in `<benchmark_name>.py` in the current directory.

> **Note**: _We rename `net -> network_config` and `scenario -> scenario_config` adapting to melissa configuration keys._


### Create json files per `ape_config` set in a specific study

#### Adding a new study
- copy-paste the `CONFIGS` dictionary from huggingface: https://huggingface.co/thuerey-group/apebench-paper/tree/main/studies
- define `for_training()`, and `for_validation()` functions alongside that perform preprocessing and the folder organization of each study. (Refer to the available scripts per study and design your experiment accordingly)
- import this new script module in the `generate_configs.py`, and add a new key in `APBENCH_STUDIES`.
- `generate_configs.py` will call `APEBENCH_STUDIES[benchmark_name]` functions.


The following commands generate configuration files for a specific study passed to the `generate_configs.py` script.

For instance, creating validation files for advection 1D case with varying difficulty can be done like this:

```bash
python3 generate_configs.py --benchmark-name=diff_adv_1d --offline
```
```bash
(apebench) abhishek@local:~/Projects/apebench_test/benchmark$ tree configs/advection_1d_varying_diff/validation/
configs/advection_1d_varying_diff/validation/
└── diff_adv
    ├── gamma_0_5
    │   └── config_diff_adv_0_5.json
    ├── gamma_10_5
    │   └── config_diff_adv_10_5.json
    └── gamma_2_5
        └── config_diff_adv_2_5.json

5 directories, 3 files
```

Similarly, create the training files per stepper type, network arch, gamma value, and the sampling type (breed, uniform, breed_mix), etc.

```bash
python3 generate_configs.py --benchmark-name=diff_adv_1d
```

```bash
(apebench) abhishek@local:~/Projects/apebench_test/benchmark$ tree configs/advection_1d_varying_diff/train
configs/advection_1d_varying_diff/train
└── diff_adv
    ├── Conv_34_0_relu
    │   ├── gamma_0_5
    │   │   ├── config_diff_adv_Conv_34_0_relu_0_5_breed.json
    │   │   ├── config_diff_adv_Conv_34_0_relu_0_5_breed_mix.json
    │   │   └── config_diff_adv_Conv_34_0_relu_0_5_uniform.json
    │   ├── gamma_10_5
    │   │   ├── config_diff_adv_Conv_34_0_relu_10_5_breed.json
    │   │   ├── config_diff_adv_Conv_34_0_relu_10_5_breed_mix.json
    │   │   └── config_diff_adv_Conv_34_0_relu_10_5_uniform.json
    │   └── gamma_2_5
    │       ├── config_diff_adv_Conv_34_0_relu_2_5_breed.json
    │       ├── config_diff_adv_Conv_34_0_relu_2_5_breed_mix.json
    │       └── config_diff_adv_Conv_34_0_relu_2_5_uniform.json
    └── FNO_12_18_4_gelu
        ├── gamma_0_5
        │   ├── config_diff_adv_FNO_12_18_4_gelu_0_5_breed.json
        │   ├── config_diff_adv_FNO_12_18_4_gelu_0_5_breed_mix.json
        │   └── config_diff_adv_FNO_12_18_4_gelu_0_5_uniform.json
        ├── gamma_10_5
        │   ├── config_diff_adv_FNO_12_18_4_gelu_10_5_breed.json
        │   ├── config_diff_adv_FNO_12_18_4_gelu_10_5_breed_mix.json
        │   └── config_diff_adv_FNO_12_18_4_gelu_10_5_uniform.json
        └── gamma_2_5
            ├── config_diff_adv_FNO_12_18_4_gelu_2_5_breed.json
            ├── config_diff_adv_FNO_12_18_4_gelu_2_5_breed_mix.json
            └── config_diff_adv_FNO_12_18_4_gelu_2_5_uniform.json

10 directories, 18 files
```

> **Note:** _Remember to set `CONFIG_FILE` environment variable on the client side as this helps the client to load scenario with the same configuration from CONFIG_FILE `json` file. This is already done while generating json files in `modify_json.py`_

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

