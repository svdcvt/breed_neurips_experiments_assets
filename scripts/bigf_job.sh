#!/bin/bash

#OAR -n melissa-study
#OAR -l /nodes=1/gpu=1,walltime=01:30:00
#OAR -p gpumodel='V100'
#OAR --stdout std/%jobid%.out
#OAR --stderr std/%jobid%.err
#OAR --project pr-melissa

source /applis/environments/singularity_env.sh

v100="cc70"
# a100="cc80" << ignore

common_path="/bettik/PROJECTS/pr-melissa/COMMON"
singularity_container="${common_path}/containers/melissa-active-sampling-with-apebench-cuda-${v100}.sif"


# run this script >> ./bigf_job.sh advection_diffusion/config_mpi.json

singularity exec \
	--nv \
	--env APEBENCH_ROOT="$HOME/apebench_test" \
	${singularity_container} \
	melissa-launcher --config_name "$1"
