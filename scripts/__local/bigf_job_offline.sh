#!/bin/bash

#OAR -n melissa-study
#OAR -l /nodes=1/gpu=1/migdevice=1,walltime=01:10:00
#OAR -t devel 
#OAR --project pr-melissa

source /applis/environments/singularity_env.sh

common_path="/bettik/PROJECTS/pr-melissa/COMMON"
singularity_container="${common_path}/containers/April23/melissa-active-sampling-with-apebench-cuda.sif"

singularity exec \
	--nv \
	--env APEBENCH_ROOT="$HOME/apebench_test" \
	${singularity_container} \
	melissa-launcher --config_name "$1"

data_path="/bettik/PROJECTS/pr-melissa/COMMON/datasets/apebench_val/ks_cons_1d/800_3waves_easy/trajectories/"
python3 $HOME/apebench_test/scripts/utils/merge_samples.py ${data_path} --quiet
