#!/bin/bash
module purge
module load profile/deeplrn
module load cuda/12.1

conda activate /leonardo_work/IscrC_EditGS/miniforge3/envs/mvp

echo "Environment ready:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python --version
echo "Working dir: $(pwd)"