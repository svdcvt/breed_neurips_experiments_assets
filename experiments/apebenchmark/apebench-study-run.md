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

`./by_folder.sh <folder-to-search> <submission-script>`


> **Note:** _It is necessary that you run one job at a time as there is some influence of SLURM daemon delays that can make study run for longer periods.
https://gitlab.inria.fr/melissa/melissa/-/issues/47_

`by_folder.sh` is written in such a way that it creates a dependency per `sbatch` call to ensure all the recursively submitted studies run one-by-one.

### Validation runs
```bash
(python-3.10.4) [uyl42ho@jean-zay3: benchmark]$ ./by_folder.sh configs/advection_1d_varying_diff/validation/diff_adv ../jz_offline_job.sh
Submitted configs/advection_1d_varying_diff/validation/diff_adv/gamma_2_5/config_diff_adv_2_5.json as Job ID 2164001
Submitted configs/advection_1d_varying_diff/validation/diff_adv/gamma_0_5/config_diff_adv_0_5.json as Job ID 2164003
Submitted configs/advection_1d_varying_diff/validation/diff_adv/gamma_10_5/config_diff_adv_10_5.json as Job ID 2164005
```

### Training runs

Running studies under `diff_burgers/` folder.

```bash
(python-3.10.4) [uyl42ho@jean-zay3: benchmark]$ ./by_folder.sh configs/broad_comparison_1d/train/diff_burgers ../jz_semig_job.sh 
Submitted configs/broad_comparison_1d/train/diff_burgers/FNO_12_18_4_gelu/config_diff_burgers_FNO_12_18_4_gelu_breed_mix.json as Job ID 2166620
Submitted configs/broad_comparison_1d/train/diff_burgers/FNO_12_18_4_gelu/config_diff_burgers_FNO_12_18_4_gelu_breed.json as Job ID 2166621
Submitted configs/broad_comparison_1d/train/diff_burgers/FNO_12_18_4_gelu/config_diff_burgers_FNO_12_18_4_gelu_uniform.json as Job ID 2166622
Submitted configs/broad_comparison_1d/train/diff_burgers/Conv_34_10_relu/config_diff_burgers_Conv_34_10_relu_breed_mix.json as Job ID 2166623
Submitted configs/broad_comparison_1d/train/diff_burgers/Conv_34_10_relu/config_diff_burgers_Conv_34_10_relu_breed.json as Job ID 2166624
Submitted configs/broad_comparison_1d/train/diff_burgers/Conv_34_10_relu/config_diff_burgers_Conv_34_10_relu_uniform.json as Job ID 2166625
```

### Cleanup
Run this to clear unnecessary folders for all the `STUDY_OUT*/` under a specified directory. This retains the tensorboard logs and the pandas pickle stored in `STUDY_OUT*/tensorboard` folder.
```bash
./cleanup.sh <folder-to-clean>
```

> **Note:** _Make sure there is no study currently running while executing cleanup._

