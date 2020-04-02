#!/bin/bash
#SBATCH -p bdw18
#SBATCH --job-name mldock-evo
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH -N 1

echo "Init..."
date
cd $SLURM_SUBMIT_DIR

module rm atp
module use /cray/css/users/dctools/modulefiles
module rm anaconda2
module load anaconda3
export PATH=/home/users/${USER}/.local/bin:${PATH}
export PYTHONUSERBASE=/home/users/${USER}/.local
export PY3PATH=/home/users/${USER}/.local/lib/python3.6/site-packages
source /cray/css/users/dctools/anaconda3/etc/profile.d/conda.sh
export PYTHONIOENCODING=utf8
export PYTHONPATH=$PY3PATH:$PYTHONPATH
export MLD_RDK_ENV_INSTALL_DIR=~/rdk_env
#conda create -y --prefix $MLD_RDK_ENV_INSTALL_DIR python=3.6
conda activate $MLD_RDK_ENV_INSTALL_DIR
#conda install -y -c conda-forge rdkit biopython scipy dask tensorflow-gpu=1.13 matplotlib
#pip install graph_nets
export CUDA_VISIBLE_DEVICES=0
export SCRATCH=/lus/scratch/${USER}

NODES=1
MLP_LATENT=32,32
MLP_LAYERS=2,2
GNN_LAYERS=8,8
FEATURES=16,16
POPSZ=250
SIGMA=0.05
MU=0.05
MODEL=/lus/scratch/jbalma/mldock_10pct_train_90pct_test_runs/model_files_87acc_90pct_test/checkpoints/model.ckpt
PROTEIN=/lus/scratch/avose/data/nhg/3RRF.nhg
INITPOP='CCCCCCCCCCCCCCCCCC'

TEMP_DIR=${SCRATCH}/temp/mldock_evo-${GNN_LAYERS}_layer-${MLP_LATENT}x${MLP_LAYERS}-nf_${FEATURES}-psz_${POPSZ}-sigma_${SIGMA}-mu_${MU}
rm -rf $TEMP_DIR
mkdir -p ${TEMP_DIR}
cp ./run_osprey.pbs ./ligand.py ./mldock_evo.py ./mldock_gnn.py ./gnn_models.py ./dataset_utils.py ./chemio.py ${TEMP_DIR}/
cd ${TEMP_DIR}
mkdir -p data
export SLURM_WORKING_DIR=${TEMP_DIR}

date
echo "Done."
echo
echo "Settings:"
pwd
ls
echo

echo "Running..."
date
time srun --pty -p bdw18 -u -N 1 -n 1 python mldock_evo.py \
    --initpop    ${INITPOP} \
    --model      ${MODEL} \
    --protein    ${PROTEIN} \
    --mlp_latent ${MLP_LATENT} \
    --mlp_layers ${MLP_LAYERS} \
    --gnn_layers ${GNN_LAYERS} \
    --popsz      ${POPSZ} \
    --sigma      ${SIGMA} \
    --mu         ${MU} \
    --features ${FEATURES} 2>&1 | tee run_osprey.log
date
echo "Done."
