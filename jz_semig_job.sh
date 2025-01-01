#!/bin/sh
#SBATCH --job-name=apebench-test
#SBATCH --output=std/%j.out
#SBATCH --error=std/%j.err
#SBATCH --time=00:15:00
#SBATCH --account=igf@cpu
#SBATCH --qos=qos_cpu-dev
##SBATCH --qos=qos_cpu-t3
#SBATCH --exclusive
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20 --cpus-per-task=1

module load cmake zeromq openmpi/4.1.5 python/3.10.4 cudnn/9.2.0.82-cuda

source $WORK/apebench/MELISSA/melissa_set_env.sh

export APEBENCH_ROOT="$WORK/apebench/test"

exec melissa-launcher --config_name "$1"
