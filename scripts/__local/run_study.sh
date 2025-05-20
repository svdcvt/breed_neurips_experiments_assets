#!/bin/bash

source /applis/environments/singularity_env.sh

common_path="/bettik/PROJECTS/pr-melissa/COMMON"
singularity_container="${common_path}/containers/April23/melissa-active-sampling-with-apebench-cuda.sif"

singularity exec \
	--nv \
	--env APEBENCH_ROOT="$HOME/apebench_test" \
	${singularity_container} \
	melissa-launcher --config_name "$1"
