#!/bin/bash
module load craype-accel-nvidia70
module rm cray-mvapich2
module rm PrgEnv-cray
#module swap cudatoolkit cudatoolkit/10.0
module load gcc/8.1.0
module use /cray/css/users/dctools/modulefiles

module rm anaconda2
module load anaconda3

export PATH=/home/users/${USER}/.local/bin:${PATH}
export PYTHONUSERBASE="/home/users/${USER}/.local"
export PY3PATH="/home/users/${USER}/.local/lib/python3.6/site-packages"
source /cray/css/users/dctools/anaconda3/etc/profile.d/conda.sh
export PYTHONIOENCODING=utf8
export PYTHONPATH="$PY3PATH:$PYTHONPATH"

export MLD_RDK_ENV_INSTALL_DIR=~/rdk_env
#conda create -y --prefix $MLD_RDK_ENV_INSTALL_DIR python=3.6
conda activate $MLD_RDK_ENV_INSTALL_DIR
#conda install -y -c conda-forge rdkit biopython scipy dask
#pip install graph_nets
