#!/bin/bash -l
#SBATCH --nodes=1 
#SBATCH --time=24:0:0 
#SBATCH --overcommit  
#SBATCH --exclusive

#srun -N 1 -n 1 --exclusive --mem=0 python sdf_to_dataset.py --sdf ./tabularResults.csv --out ./data_covid19 

source ./setup_env.sh
rm -rf ./obsolete
rm -rf ./data-6vsb-bindingdb-fda-all
mkdir -p ./data-6vsb-bindingdb-fda-all
#wget http://www.bindingdb.org/bind/drugatfda_BindingDBf2D.sdf 
export OMP_NUM_THREADS=18

time srun -N 1 -n 1 -c 18 --exclusive --mem=0 -p bdw18 -t 24:00:00 -u python -u pdb_list_to_testset.py --pdb_id '6vsb' --out ./data-6vsb-bindingdb-fda-all --sdf ./drugatfda_BindingDBf2D.sdf 2>&1 |& tee pharml-preproc-covid19-6vsb-fda-all.out



