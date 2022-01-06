#!/bin/bash

# Die on failure
set -e

if [ -n "$CUDA_VISIBLE_DEVICES" ];
then
	NUM_GPUS=$(echo $CUDA_VISIBLE_DEVICES | tr "," "\n" | wc -l)
else
	NUM_GPUS=$(nvidia-smi --list-gpus | wc -l)
fi

if [ ${NUM_GPUS} -gt 1 ]
then
	DIST=" -m torch.distributed.launch --nproc_per_node ${NUM_GPUS} --master_port $(($RANDOM+1024)) --use_env "
fi

if [ ${SLURM_JOB_NAME} ]
then
        echo ${SLURM_JOB_NAME} ${SLURM_JOB_ID}
        SRUN="srun"
else
        echo "Running locally"
fi

PYTHON_CMD="${SRUN} python ${DIST}"
