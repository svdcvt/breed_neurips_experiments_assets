#!/bin/bash

source /applis/environments/singularity_env.sh

v100="cc70"
# a100="cc80" << ignore

common_path="/bettik/PROJECTS/pr-melissa/COMMON"
# singularity_container="${common_path}/containers/archive/Feb28/melissa-active-sampling-with-apebench-cuda-${v100}.sif"
singularity_container="${common_path}/containers/April18/melissa-active-sampling-with-torch-apebench-cuda.sif"


# run this script >> ./bigf_job.sh advection_diffusion/config_mpi.json

singularity exec \
	--nv \
	--env APEBENCH_ROOT="$HOME/apebench_test" \
	${singularity_container} \
	melissa-launcher --config_name "$1"
