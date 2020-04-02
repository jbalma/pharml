#!/bin/bash -l
#SBATCH --nodes=1 
#SBATCH --time=24:0:0 
#SBATCH --overcommit  
#SBATCH --exclusive

source ./setup_env.sh

srun -N 1 -n 1 --exclusive --mem=0 python sdf_to_dataset.py --sdf /lus/scratch/${USER}/mldock/BindingDB_All_terse_2D.sdf
