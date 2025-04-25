#!/bin/bash
cd "$(dirname "$0")"
timeout ${1:-300} nvidia-smi -i 0 --query-gpu=timestamp,utilization.gpu,utilization.memory --format=csv -l 1 > gpu_stats.csv