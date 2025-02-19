#!/bin/bash

#OAR --project test
#OAR --stdout=std/%jobid%.out
#OAR --stderr=std/%jobid%.err


##OAR -l /nodes=1/gpu=1
##OAR -p "gpumodel='V100'"

source /applis/environments/singularity_env.sh

v100="cc70"
a100="cc80"

common_path="/bettik/PROJECTS/pr-melissa/COMMON"
singularity_container="${common_path}/containers/melissa-with-apebench-cuda-${v100}.sif"

singularity exec \
	--nv \
	--env JAX_PLATFORMS="cpu" \
	--env APEBENCH_ROOT="$HOME/apebench_test" \
	${singularity_container} \
	melissa-launcher --config_name "$1"
