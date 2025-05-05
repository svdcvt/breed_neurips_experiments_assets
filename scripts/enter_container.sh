#!/bin/bash

v100="cc70"
# a100="cc80" << ignore

common_path="/bettik/PROJECTS/pr-melissa/COMMON"
# singularity_container="${common_path}/containers/archive/Feb28/melissa-active-sampling-with-apebench-cuda-${v100}.sif"
# singularity_container="${common_path}/containers/April18/melissa-active-sampling-with-torch-apebench-cuda.sif"
singularity_container="${common_path}/containers/April23/melissa-active-sampling-with-apebench-cuda.sif"

singularity shell \
	--nv \
	--bind /bettik:/bettik \
	--env APEBENCH_ROOT="$HOME/apebench_test" \
	${singularity_container}
