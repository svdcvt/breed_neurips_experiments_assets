#!/bin/bash

v100="cc70"
# a100="cc80" << ignore

common_path="/bettik/PROJECTS/pr-melissa/COMMON"
singularity_container="${common_path}/containers/melissa-active-sampling-with-apebench-cuda-${v100}.sif"

singularity shell \
	--nv \
	--env APEBENCH_ROOT="$HOME/apebench_test" \
	${singularity_container}
