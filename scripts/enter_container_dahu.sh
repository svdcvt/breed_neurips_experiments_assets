#!/bin/bash

singularity shell --bind /bettik:/bettik --env APEBENCH_ROOT="$HOME/2024/apebench_test" /bettik/PROJECTS/pr-melissa/COMMON/containers/April23/melissa-active-sampling-with-apebench-cuda.sif
