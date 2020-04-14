#!/bin/bash


#module load PrgEnv-cray
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

#/cray/css/users/jbalma/Applications/OpenMPI-3/ompi-cuda92-osprey
#export MPI_PATH=/cray/css/users/jbalma/Applications/OpenMPI-3/ompi-cuda10-osprey
export MPI_PATH=/cray/css/users/jbalma/Applications/OpenMPI-4/ompi-gcc72-cuda10-osprey
#export MPI_PATH=/cray/css/users/jbalma/Applications/OpenMPI-3/ompi-cuda92-osprey
#export MPI_PATH=/lus/scratch/pjm/ompi-cuda
#export MPI_PATH=/lus/scratch/pjm/ompi3.1.2-cuda
#export MPI_PATH=/lus/scratch/pjm/ompi
export OPAL_PREFIX=${MPI_PATH}
export PATH=${MPI_PATH}/bin:${PATH}

##export cc=${MPI_PATH}/bin/mpicc
#export CC=${MPI_PATH}/bin/mpicxx
#export CUDNN_PATH=/cray/css/users/jbalma/Tools/CuDNN/cudnn-9.2-v721/cuda
export CUDNN_PATH=/cray/css/users/jbalma/Tools/CuDNN/cudnn-10.0-v742/cuda
#export CUDNN_PATH=/cray/css/users/jbalma/Tools/CuDNN/cudnn-10.1-v762/cuda
export TOOLS=/cray/css/users/jbalma/Tools
#export CUDATOOLKIT_HOME=/global/opt/nvidia/cudatoolkit/9.2
export CUDATOOLKIT_HOME=/global/opt/nvidia/cudatoolkit/10.0
export LD_LIBRARY_PATH=${MPI_PATH}/lib:${CUDNN_PATH}/lib64:${CUDATOOLKIT_HOME}/lib64:${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}


export MLD_RDK_ENV_INSTALL_DIR=~/cuda10_env_pharml
#export PYTHONUSERBASE="/home/users/${USER}/.local"
#export PY3PATH="/home/users/${USER}/.local/lib/python3.6/site-packages"
#source /cray/css/users/dctools/anaconda3/etc/profile.d/conda.sh
export PYTHONIOENCODING=utf8
#export PYTHONPATH="$PY3PATH:$PYTHONPATH"

#conda create -y --prefix $MLD_RDK_ENV_INSTALL_DIR python=3.6 cudatoolkit=10.0 cudnn
#conda activate $MLD_RDK_ENV_INSTALL_DIR/
#export PATH=${MLD_RDK_ENV_INSTALL_DIR}/bin:/home/users/${USER}/.local/bin:${PATH}
#conda install -y -c conda-forge rdkit biopython scipy dask
#conda install -y -c anaconda pip
#pip uninstall tensorflow tensorflow-gpu tensorflow-probability
#pip uninstall tensorflow tensorflow-gpu
#pip uninstall tf-nightly-gpu tf-estimator-nightly
#pip install tf-nightly-gpu graph_nets dm-sonnet==1.25 tensorflow-probability matplotlib
#pip install tensorflow_gpu==1.15
#pip install graph_nets dm-sonnet==1.25 tensorflow-probability matplotlib
#When this finishes, run ./install_horovod_python3_cs.sh
#conda deactivate

#source ./install_horovod_python3_cs.sh


