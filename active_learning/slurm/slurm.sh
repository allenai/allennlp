#!/bin/bash
#
#SBATCH --mem=80000
#SBATCH --job-name=al-bidaf
#SBATCH --partition=m40-long
#SBATCH --output=al-%A.out
#SBATCH --error=al-%A.err
#SBATCH --gres=gpu:1


# Log what we're running and where.
echo $SLURM_JOBID - `hostname` >> ~/slurm-jobs.txt

module purge
module load python/3.6.1
module load cuda80/blas/8.0.44
module load cuda80/fft/8.0.44
module load cuda80/nsight/8.0.44
module load cuda80/profiler/8.0.44
module load cuda80/toolkit/8.0.44

## Change this line so that it points to your bidaf github folder
cd /home/usaxena/work/s18/696/allennlp/active_learning

python make_data.py --percent 2 --gpu 1