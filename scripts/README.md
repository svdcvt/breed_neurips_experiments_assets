# Scripts 

The workflow involves:

1. [Creating configurations](#configuration-management-scripts) for experiments (training online and creating validation dataset(s) offline)
2. [Running](#job-management-scripts):
   - offline study to generate validation dataset(s)
   - online training experiments
3. [Visualizing results](#visualisation)

See the [complete workflow example](#complete-workflow-example) for a step-by-step guide.

Before anything, set environment variables:
```bash
export REPO_ROOT=/path/to/this/repo
export DATASET_ROOT=/path/to/save/validation/datasets
```

## Configuration Management Scripts

### [`create_configs_set.py`](./sets/create_configs_set.py)

**Purpose**: Creates configuration sets for experiments from templates. 
Inside the script, common parameters for an experiment set are defined ([line 47](./sets/create_configs_set.py#L47)). Then, we can iterate over changing parameters for the set ([line 97](./sets/create_configs_set.py#L97)).
For experiment A and B, these parts are described in [here](./sets/create_config_studies_.md).
Running the script will create all config files (both online and offline) in structured directories, which then can be ran with `melissa-launcher`.

**Usage**:
```bash
python create_configs_set.py --study-directory PATH_TO_STUDY_DIR --subdir SUBDIR_NAME
```

**Example**:
```bash
python create_configs_set.py --study-directory experiments --subdir set
```
This script creates configs based on the following `.json` file.

### [`default_configs.json`](./sets/default_configs.json)

**Purpose**: Default configuration templates for online and offline modes.

Edit this file to change default settings for experiments. The templates include server , clients, launcher configurations.

### [`make_config.py`](./sets/make_config.py)

**Purpose**: Core configuration generator containing classes for different aspects of the experiment setup.

This script is imported by `create_configs_set.py` and provides the data structures used to generate configurations:
- `ScenarioConfig`: PDE scenario settings (wave counts, coefficients, PDE type)
- `DLConfig`: Deep learning model settings (architecture, learning rate)
- `MelissaConfig`: Training infrastructure settings (buffer size, study size, compute budget)
- `ActiveSamplingConfig`: Parameter sampling regimes
- `StudyConfig`: Combines all configs into study setup (makes nice paths and constructs configs)

### [`create_config_studies_.md`](./sets/create_config_studies_.md)

**Purpose**: Documentation and examples for creating study configurations.

Refer to this file for guidance on setting up different types of experiments.


## Job Management Scripts

### [`make_job_set.sh`](./sets/make_job_set.sh)

**Purpose**: This generates cluster job submission scripts for each matching configuration file. Use when you want to start a batch of training experiments. 


**Usage**:
```bash
./make_job_set.sh PATH_TO_CONFIGS_FOLDER NUMBER_SCRIPTS MIN_PER_STUDY
```

**Example**:
```bash
./make_job_set.sh ../../experiments/set 14 70
```

This will iterate over all the configs in the folder for experiment A (14 PDEs x 6 methods) and create 14 scripts, each running 6 studies (one after another), each script will request 6*70 minutes of resources. You have to change [line 92](./sets/make_job_set.sh#L92) according to your cluster scheduler.
Created script can be then used to submit a job on the cluster. If there are many scripts, possibly filtered subset, and they have to be submitted all together, use the next script.

### [`make_offline_set.sh`](./sets/make_offline_set.sh)

**Purpose**: Almost the same as `make_job_set.sh` but for offline validation creation. The scripts are created in the script directory.
Change [this line](./sets/make_offline_set.sh#L89) accordingly.
Submit these scripts manually and also combine created trajectories in one file with use of [`merge_samples.py`](#merge_samplespy)

### [`meta_make_job_set.sh`](./sets/meta_make_job_set.sh)

**Purpose**: Creates meta-scripts that manage multiple job sets, useful for large experiment runs. It will run `make_job_set.sh` for each recursive subfolder in the provided folder name and its path contains provided filter match

**Usage**:
```bash
./meta_make_job_set.sh PATH_TO_CONFIGS_FOLDER FILTER MIN_PER_STUDY
```

**Example**:
```bash
./meta_make_job_set.sh ../../experiments/set ks__cons 70
```

Here we want to run only `ks_cons` studies (4 variations x 6 methods).
This will create 4 job submission scripts, each running 6 studies, with request of 6*70 minutes. As well, a script `job_set_0.sh` will be created, that can be run to to do submission of all created scripts.
You have to change [this line](./sets/meta_make_job_set.sh#L86) according to your cluster.

If you did a mistake creating the scripts, run the same line but with delete flag:
```bash
./meta_make_job_set.sh ../../experiments/set ks__cons 70 del
```
It will delete the scripts you've just created.


## Data Management Scripts

### [`merge_samples.py`](./utils/merge_samples.py)

**Purpose**: Combines multiple simulation samples into a single dataset file.

**Usage**:
```bash
python merge_samples.py /path/to/trajectories/directory --test
```

Run this after offline dataset generation to consolidate individual simulation files into a single validation dataset. First, do it with `--test` flag to check if the merging works. If it does, run it again without the flag to create the final dataset. It will delete sample files during merging.


### [`find_validation_ids.py`](./utils/find_validation_ids.py)

**Purpose**: Identifies useful validation dataset samples by analyzing trajectory differences.

**Usage**:
```bash
python find_validation_ids.py --subdir-match "diff_*" --num 5
```
This script finds `num` representative samples in validation datasets matching `subdir-match` for visualization. Run after generating validation datasets.
It will create a csv file with the folder/pde name, validation ids, and metrics for these ids. The metric is variation of variations of differences between consecutive timesteps in the sample. The lower the metric, the more constant the trajectory is, and therefore easier.
We use these ids for prediction visualisations.


### [`study_size_recommendations.py`](./utils/study_size_recommendations.py)

**Purpose**: Calculates memory requirements and recommends dataset sizes.

**Usage**:
```bash
python study_size_recommendations.py --space-points 800 --time-points 75 \
    --val-samples 1000 --val-time-points 100 --precision 32 --memory 30
```

Use before running experiments to determine appropriate buffer sizes and sample counts based on available memory.

## PDE Configuration Data

### pde_set.csv, pde_set_few.csv, pde_set_names.csv

**Purpose**: Contains PDE parameter configurations for experiments.

These files define the PDE types, coefficients, and horizon values for experiments:
- pde_set_few.csv: A small set for experiment B
- pde_set.csv: A comprehensive set covering many PDE configurations for experiment A

Use these to create new PDE configurations for experiments.

### [`exponax_solvers_investigation.ipynb`](./utils/exponax_solvers_investigation.ipynb)

**Purpose**: Jupyter notebook for analyzing and comparing different solvers, coefficients, IC variations with sliders. Helps to find interesting PDE configurations for experiments.

## Visualisation

### [`plot_client_parameters.py`](./utils/plot_client_parameters.py)

**Purpose**: Visualizes parameter distributions across created parameters for clients during the training.

**Usage**:
```bash
python plot_client_parameters.py --input-dir /path/to/experiment/client_scripts \
    --validation-path /path/to/validation/all_parameters.npy 
```

Run after experiments to visualize how parameter sampling evolved during training.


### [`utils/create_all_plot_scripts.sh`](./utils/create_all_plot_scripts.sh) and [`utils/plot_some.sh`](./utils/plot_some.sh)

**Purpose**: Generate scripts for creating visualization plots and plotting specific experiment results. It uses [`plot_model_predictions.py`](../common/plot_model_predictions.py), which inference the model with checkpoint weights and creates plots for the predictions. These scripts are designed to be run on CPU nodes.
Change manually lines 3-5, 11, 18, 50.



# Complete Workflow Example

0. **Set up environment**:
   Make sure to set the environment variables `REPO_ROOT` and `DATASET_ROOT` as described above.
   ```bash
   export REPO_ROOT=/path/to/this/repo
   export DATASET_ROOT=/path/to/save/validation/datasets
   ```
   Also, make sure to install dependencies as described in []( or use the container.

1. **Create configuration files**:
Change the parameters in the script to your needs. Then run:
   ```bash
   python $REPO_ROOT/scripts/utils/create_configs_set.py --study-directory $REPO_ROOT/experiments --subdir myset
   ```

2. **Generate validation datasets**:
   ```bash
   $REPO_ROOT/scripts/sets/make_offline_set.sh $REPO_ROOT/experiments/myset 14 45
   find -name ./offline_job_set_*.sh -exec CLUSTER_SUBMIT {} \;
   ```

3. **Merge validation samples**:
   ```bash
   find -name $DATASET_ROOT/*/trajectories/ -type d -exec python $REPO_ROOT/scripts/utils/merge_samples.py {} --test \;
   find -name $DATASET_ROOT/*/trajectories/ -type d -exec python $REPO_ROOT/scripts/utils/merge_samples.py {} \;
   ```

5. **Run training experiments**:
   ```bash
   $REPO_ROOT/scripts/sets/meta_make_job_set.sh $REPO_ROOT/experiments/myset diff_ 70
   $REPO_ROOT/experiments/myset/job_set_meta_diff_.sh
   ```

6. **Visualise/analyse results**:
   ```bash
   python $REPO_ROOT/scripts/utils/plot_client_parameters.py --input-dir $REPO_ROOT/experiments/myset/diff_kdv__2w_x10_easier_max1_1d_x5/client_scripts \
       --validation-path $DATASET_ROOT/diff_kdv__2w_x10_easier_max1_1d_x5/trajectories/all_parameters.npy
   
   python $REPO_ROOT/scripts/utils/find_validation_ids.py --subdir-match "diff_" --num 5
   ```
   
   adapt the parameters in the script to your needs and ensure the paths are correct.

   ```bash
   $REPO_ROOT/utils/create_all_plot_scripts.sh
   ```

