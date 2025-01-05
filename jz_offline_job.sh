#!/bin/sh

#SBATCH --job-name=validation-generation
#SBATCH --output=std/%j.validation.out
#SBATCH --error=std/%j.validation.err
#SBATCH --time=00:10:00
#SBATCH --account=igf@cpu
#SBATCH --qos=qos_cpu-dev
#SBATCH --ntasks=1 --cpus-per-task=10
#SBATCH hetjob
#SBATCH --ntasks-per-node=20 --cpus-per-task=1

module load cmake zeromq openmpi/4.1.5 python/3.10.4 cudnn/9.2.0.82-cuda

source $WORK/apebench/MELISSA/melissa_set_env.sh

export APEBENCH_ROOT="$WORK/apebench"
export ADVECTION_DIR="$APEBENCH_ROOT/test/advection"

exec melissa-launcher --config_name "$1"
