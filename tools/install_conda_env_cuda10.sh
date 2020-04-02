#!/bin/bash -l

module load PrgEnv-cray
module load craype-accel-nvidia70
module rm cray-mvapich2/2.2rc1
module rm PrgEnv-cray
module rm cudatoolkit
module load cudatoolkit/10.0
module rm gcc
module load gcc/7.2.0
module use /cray/css/users/dctools/modulefiles
module rm anaconda2
module load anaconda3
module list

export MPI_PATH=/cray/css/users/&{USER}/Applications/OpenMPI-4/ompi-gcc72-cuda10-osprey
export OPAL_PREFIX=${MPI_PATH}
export PATH=${MPI_PATH}/bin:${PATH}

export cc=${MPI_PATH}/bin/mpicc
export CC=${MPI_PATH}/bin/mpicxx
export CUDNN_PATH=/cray/css/users/&{USER}/Tools/CuDNN/cudnn-10.0-v742/cuda
export TOOLS=/cray/css/users/&{USER}/Tools
export CUDATOOLKIT_HOME=/global/opt/nvidia/cudatoolkit/10.0
export LD_LIBRARY_PATH=${MPI_PATH}/lib:${CUDNN_PATH}/lib64:${CUDATOOLKIT_HOME}/lib64:${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}

export PATH=/home/users/${USER}/.local/bin:${PATH}
export PYTHONUSERBASE="/home/users/${USER}/.local"
export PY3PATH="/home/users/${USER}/.local/lib/python3.6/site-packages"
source /cray/css/users/dctools/anaconda3/etc/profile.d/conda.sh
export PYTHONIOENCODING=utf8
export PYTHONPATH="$PY3PATH:$PYTHONPATH"

export MLD_RDK_ENV_INSTALL_DIR=~/cuda10_env
conda create -y --prefix $MLD_RDK_ENV_INSTALL_DIR python=3.6 cudatoolkit=10.0 cudnn
source activate $MLD_RDK_ENV_INSTALL_DIR
conda install -y -c conda-forge rdkit biopython scipy dask
pip install tensorflow-gpu==1.15 graph_nets dm-sonnet==1.25 tensorflow-probability

#When this finishes, run ./install_horovod_python3_cs.sh



