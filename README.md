
# Supplementary Materials Instructions


In order to follow Open Science principles, we want to give the code, the experiments and its assets in as complete as possible form. 
However, to respect the double-blind review process, we have to anonymise everything.

We fully anonymised the codebase itself, i.e., the algorithm code, scripts that create all configuration files, scripts that run the experiments, scripts that process the results of the experiments and create the plots that are included in the paper.
Conducted and presented in the paper experiments can be reproduced almost exactly[^1] with these scripts, although, it can take up to ~150 compute hours (80% GPU hours) and ~40+ GB storage.
We understand it is an immense effort, even more, it is an additional carbon footprint of computational research.
Therefore, we want to include all our experimental assets. 

Some of the files are large, some of them do not participate in the creation of the paper figures. But the whole experimental workflow is sequential, so we come up with the system of levels of the assets that can be shared. The levels are ordered by the amount of data and the amount of effort to reproduce the experiments. 

It is left to the reviewers' decision which option suits them better.


|Level| Includes| Produces| Source |
|---        |---|---|---|
| **Code** | <ul><li>Algorithms codes</li><li>Experiments codes</li><li>Scripts "end-to-end": <ul><li>generating config. files</li><li>running experiments</li><li>processing exp. results</li><li>creating plots</li></ul></li><li>Part of results files:<ul> <li>combined `.csv` A, B</li></ul></li></ul>  | <ul><li>Table A statistics <ul><li>for checkpoints 1k, 2k, **3k**, 4k</li></ul></li><li>Figure 3<ul><li>for all PDE cases</li><li>**rollout** and other metrics from Table A</li></ul></li><li>Figure 4</li></ul> Can produce the same configs used for ExpA and ExpB, and to create validation datasets.| **This repository.** <br> Code: <ul><li>[Common](./common/README.md)</li><li>[Scripts](./scripts/README.md)</li></ul> Data and plotting: <ul><li>Main experiments (ExpA): [combined `.csv`](./validation_results/validation_results_decay/combined_validation_results.csv)</li><li>Random seed experiments (ExpB): [combined `.csv`](./validation_results/random_seed_tensorboard_results.csv)</li><li>Plotting these results: [ipynb](./validation_results/csv_to_tab_plot.ipynb)</li></ul>  Fork of Melissa with the algorithm: [url](url |
|**+data**  |  For ExpA, per PDE, per method, per model checkpoint: <ul><li>`.npy` data for plots</li><li>`.csv` metrics</li><li>`.pdf` plots</li></ul> |  Combined `.csv` , and for any PDE/method, available chkpt:<ul><li>Figure 1ab</li><li>Figure 2</li></ul>   | Download [here](url (1.1GB) and unzip into `validation_results` directory |
|**+models, +valset**  |  For ExpA, per PDE, per method: <ul><li>model weights chkpt every 1k iterations</li></ul> Full validation datasets for all PDEs| <ul><li>`.npy` data for plots</li><li>`.csv` metrics</li><li>`.pdf` plots</li></ul> | Download chkpts [here](url and unzip[^2] into `experiments` directory, can be combined with level "+tensorboard". <br> Download val. datasets [here](url (~30GB). |
|**+tensorboard**  |  For ExpB, per PDE, per method:<ul><li>`tensorboard` event file</li></ul> | <ul><li> ExpB combined `.csv` B</li><li> can see all other statistics, e.g., training loss  | Download [here](url , unzip[^2] into `experiments` directory, can be combined with level "+models" |
|**Fullest**  |  Copy of the whole workspace. <br> All configs, logs, Melissa scripts (<span style="color:red">not anonymised</span>), and as well data, model chkpts, tensorboards, .... |   | Download [here](url, the archive already contains everything from the previous levels. |
|**Semi**  |  Level "Code" + anonymised configs (view only)|    | See here:  |

1*: While all the models and samples parameters are fixed by the random seed, the order of execution of the solvers and the batch collection is not deterministic, which can affect the results, however, it should not be significant. 
2*: The right way to unzip the archive to preserve directories structure: `unzip TODO`

![image](./experiment_workflow.png)
Figure 1: The workflow of the experiments. The arrows is code execution. The figure names are "type" of a figure, same as in the paper, and figure images connected to them are all possible figures like this, which are included in the supplement materials (and in `+data`).
Legend:
- METRICs (calculated over the full validation dataset):
    - full rollout nRMSE average,
    - one-to-one MSE mean, max, std, median.
- METHODs: broad, precise, mixed, soft, uniform, static (described in the paper and can be seen here: [`make_config.py`](./scripts/sets/make_config.py#L408))
- PDEs: 14 cases, see [table](scripts/utils/pde_set_names.csv).
- CHKPTs: model weights checkpoints available, usually 1k-2k-3k-(4k).


## Container Usage Instructions

We provide a Singularity container, to which all the executable scripts are referring to. It was used for all the experiments. The container can be downloaded from here: [link](link
The container is expected (by the scripts) to be located in the root of the repository. You can as well enter the container with the following command:
```bash
singularity shell --nv --env REPO_ROOT=$REPO_ROOT <path_to_container>
```
and run any study with the command:
```bash
melissa-launcher --config <path_to_config>
```

<span style="color:red"> DISCLAIMER: </span> it is **impossible** to anonymise the container, as it was built by the authors, and it has paths that include username(s) of the author(s). Given the rule that the reviewers are not supposed to intentionally try to **identify authors** with any given clues, we believe it is possible to use the container without breaking the double-blind review process. In the end, the container is given as an easy way to reproduce the environment of the experiments. 

We provide the instructions to install dependencies without the container, but we do not guarantee the exact environment reproducibility.

## Installation Instructions

### Local Melissa and JAX/APEBench with `conda` for GPU

Before installing, make sure `LD_LIBRARY_PATH` is not set.


#### Option 1 (No CUDA present)
There is a `jax[cuda12]`, which should install its own CUDA toolkit and cuDNN version from Python wheels.

```bash
conda create -n apebench python=3.10
conda activate apebench
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade "jax[cuda12]"
python3 -m pip install apebench
```

#### Option 2 (CUDA present)
If option 1 does not work, add `cuda cudnn` to `conda create` and let `jax` installation use local versions, like this:

```bash
conda create -n apebench python=3.10 cuda cudnn
conda activate apebench
python3 -m pip install --upgrade pip
python3 -m pip install --upgrade "jax[cuda12_local]"
python3 -m pip install apebench
```

**NOTE:** `jax[cuda12_local]` should also work if you have CUDA packages already installed on the machine. Making external CUDA work seamlessly with the conda environment might be problematic. In that case, use `python3-venv` environment instead.

---

To check that GPU is available:
```bash
(apebench) user@user:~/path/to/repo$ python3
Python 3.10.16 (main, XXX XX 202X, XX:XX:XX) [GCC 11.2.0] on linux
Type "help", "copyright", "credits" or "license" for more information.
>>> import jax
>>> jax.devices()
[CudaDevice(id=0)]
```

Set `XLA_FLAGS` for CUDA library search path to find `nvvm/libdevices.so`,
```bash
export XLA_FLAGS="--xla_gpu_cuda_data_dir=$CONDA_PREFIX"
```

Test a toy training using APEBench,
```bash
python3 sample/hello.py
```

Melissa's installation should be done outside the conda environment. Therefore, execute,

```bash
python3 -m pip install -r melissa_requirements.txt
curl -L -o /tmp/melissa.zip <url-for-melissa.zip> && unzip /tmp/melissa.zip -d $HOME/melissa
./build_and_install_melissa.sh $HOME/melissa
```

These commands will install Melissa module inside `$HOME/melisssa/install`. Users are free to change the location for unzipping.

**NOTE:** It is important to initialize Melissa by executing `source $HOME/melissa/melissa_set_env.sh`
before making a run. The script is responsible for setting environment variables exposing melissa binaries, libraries, etc.

#### Test Melissa
```bash
source $HOME/melissa/melissa_set_env.sh
melissa-launcher --print-options
```
and run any study with the command:
```bash
melissa-launcher --config <path_to_config>
```

