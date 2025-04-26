#!/bin/bash

#OAR -n melissa-study
#OAR -l /nodes=1/gpu=2,walltime=00:35:00
#OAR -p gpumodel='V100'
#OAR --project pr-melissa

source /applis/environments/singularity_env.sh

common_path="/bettik/PROJECTS/pr-melissa/COMMON"
# singularity_container="${common_path}/containers/archive/Feb28/melissa-active-sampling-with-apebench-cuda-${v100}.sif"
# singularity_container="${common_path}/containers/April18/melissa-active-sampling-with-torch-apebench-cuda.sif"
singularity_container="${common_path}/containers/April23/melissa-active-sampling-with-apebench-cuda.sif"


# run this script >> ./bigf_job.sh advection_diffusion/config_mpi.json

singularity exec \
	--nv \
	--env APEBENCH_ROOT="$HOME/apebench_test" \
	${singularity_container} \
	melissa-launcher --config_name "$1"
