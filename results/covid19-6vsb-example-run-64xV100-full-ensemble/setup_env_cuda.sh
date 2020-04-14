#!/bin/bash
module load craype-accel-nvidia70
module rm cray-mvapich2/2.2rc1
module rm PrgEnv-cray
module rm cudatoolkit
module load cudatoolkit/9.0.176
module use /cray/css/users/dctools/modulefiles
module rm anaconda2
module load anaconda3
module load gcc/7.2.0
module list
export MPI_PATH=/cray/css/users/jbalma/Applications/OpenMPI-4/ompi-cuda90-osprey
export OPAL_PREFIX=${MPI_PATH}
export PATH=${MPI_PATH}/bin:${PATH}

#export cc=${MPI_PATH}/bin/mpicc
#export CC=${MPI_PATH}/bin/mpicxx
export CUDNN_PATH=/cray/css/users/jbalma/Tools/CuDNN/cudnn-9.0-v71/cuda
export TOOLS=/cray/css/users/jbalma/Tools
export CUDATOOLKIT_HOME=/global/opt/nvidia/cudatoolkit/9.0.176
export LD_LIBRARY_PATH=${MPI_PATH}/lib:${CUDNN_PATH}/lib64:${CUDATOOLKIT_HOME}/lib64:${LD_LIBRARY_PATH}

conda deactivate

export PATH=/home/users/${USER}/.local/bin:${PATH}
export PYTHONUSERBASE="/home/users/${USER}/.local"
export PY3PATH="/home/users/${USER}/.local/lib/python3.6/site-packages"
source /cray/css/users/dctools/anaconda3/etc/profile.d/conda.sh
export PYTHONIOENCODING=utf8
export PYTHONPATH="$PY3PATH:$PYTHONPATH"

export MLD_RDK_ENV_INSTALL_DIR=~/cuda90_env
conda create -y --prefix $MLD_RDK_ENV_INSTALL_DIR python=3.6 cudatoolkit==9.0 cudnn
source activate $MLD_RDK_ENV_INSTALL_DIR
conda activate $MLD_RDK_ENV_INSTALL_DIR
conda install -y -c conda-forge rdkit biopython scipy dask
pip install protobuf==3.9.1
pip install tensorflow-gpu==1.12 graph_nets dm-sonnet==1.25 tensorflow-probability==0.5
pip install matplotlib
source ../tools/install_horovod_python3_cs.sh



