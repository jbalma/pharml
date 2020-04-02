#!/bin/bash -l
#SBATCH --nodes=1 
#SBATCH --time=24:0:0 
#SBATCH --overcommit  
#SBATCH --exclusive

#srun -N 1 -n 1 --exclusive --mem=0 python sdf_to_dataset.py --sdf ./tabularResults.csv --out ./data_covid19 
source ./setup_env.sh
rm -rf ./obsolete
#wget http://www.bindingdb.org/bind/drugatfda_BindingDBf2D.sdf 

srun -N 1 -n 1 --exclusive --mem=0 -t 24:00:00 python pdb_list_to_testset.py --pdb_id '6VSB' --out ./data --sdf ./drugatfda_BindingDBf2D.sdf 2>&1 |& tee pharml-preproc-covid19.out



