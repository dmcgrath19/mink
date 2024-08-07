#!/bin/bash

#$ -N mink-py-m

#$ -o /exports/eddie/scratch/s2558433/job_runs/mink-$JOB_ID.log
#$ -e /exports/eddie/scratch/s2558433/job_runs/mink-$JOB_ID.err
#$ -cwd
#$ -q gpu
#$ -l rl9=true
#$ -pe gpu-a100 1
#$ -l h_vmem=300G
#$ -l h_rt=24:00:00
#$ -m bea -M s2558433@ed.ac.uk 

export HF_HOME="/exports/eddie/scratch/s2558433/.cache/huggingface_cache"
export TRANSFORMERS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/transformers"
export HF_DATASETS_CACHE="/exports/eddie/scratch/s2558433/.cache/huggingface_cache/datasets"
export PIP_CACHE_DIR="/exports/eddie/scratch/s2558433/.cache/pip"
export CONDA_PKGS_DIRS="/exports/eddie/scratch/s2558433/.cache/conda_pkgs"

export MIMIR_CACHE_PATH="/exports/eddie/scratch/s2558433/.cache/mimir_cache"
export MIMIR_DATA_SOURCE="/exports/eddie/scratch/s2558433/.cache/mimir_ds_cache"


. /etc/profile.d/modules.sh
module unload cuda
module load cuda/12.1.1


source /exports/eddie/scratch/s2558433/miniconda3/etc/profile.d/conda.sh
module load anaconda
conda activate mink
cd /exports/eddie/scratch/s2558433/mink/

# python run_neighbor.py --model state-spaces/mamba-130m-hf --dataset WikiMIA_length32_paraphrased
# python run_neighbor.py --model state-spaces/mamba-130m-hf --dataset WikiMIA_length64_paraphrased
# python run_neighbor.py --model state-spaces/mamba-130m-hf --dataset WikiMIA_length128_paraphrased

python run.py --model EleutherAI/pythia-1b --dataset WikiMIA_length32
python run_ref.py --model EleutherAI/pythia-1b  --dataset WikiMIA_length32

# python run_neighbor.py --model state-spaces/mamba-130m-hf --dataset WikiMIA_length64
# python run_neighbor.py --model state-spaces/mamba-130m-hf --dataset WikiMIA_length128