#!/bin/bash
#This script assumes you have already sourced install_conda_env_cuda.sh
#the conda environment build script which creates ~/cuda90_env 
#and installs all needed dependencies there for running PharML.Bind
#source ./setup_env_cuda90.sh
source ./setup_env_cuda10_covi19.sh

export MPI_CC=${MPI_PATH}/bin/mpicc
export MPI_CXX=${MPI_PATH}/bin/mpicxx

#export MLD_RDK_ENV_INSTALL_DIR=~/cuda10_env
source activate $MLD_RDK_ENV_INSTALL_DIR

which python
#pip3 install protobuf==3.9.1

#Might need to uninstall and reinstall tensorflow-gpu with the --user flag if this doesn't work
#pip uninstall tf-nightly-gpu tf-estimator-nightly tensorflow-gpu tensorflow tensorflow-estimator horovod tensorflow_estimator
#pip3 install tensorflow-gpu==1.12 --user
#pip install tf-nightly-gpu
#pip install tensorflow-estimator==1.14.0
pip uninstall horovod
#pip3 uninstall horovod
#If you want to use PyTorch with Horovod, uncomment this
#pip3 install torch torchvision --upgrade --user


#This option might be needed depending on tensorflow, pytorch versions
#-D_GLIBCXX_USE_CXX11_ABI=0

echo $CUDATOOLKIT_HOME
which gcc 
which python

#Vanilla Cluster with OpenMPI
CXX=$MPI_CXX CC=$MPI_CC HOROVOD_CUDA_HOME=${CUDATOOLKIT_HOME} HOROVOD_MPICXX_SHOW="mpicxx -show" HOROVOD_MPI_HOME=${MPI_PATH} HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip install --global-option=build_ext --global-option="-I ${CUDATOOLKIT_HOME}/include" -v --no-cache-dir horovod

#Cray XC
#CC=cc CXX=CC HOROVOD_MPICXX_SHOW="CC --cray-print-opts=all" HOROVOD_CUDA_HOME=$CUDATOOLKIT_HOME HOROVOD_MPICXX_SHOW="CC --cray-print-opts=all" HOROVOD_GPU_ALLREDUCE=MPI HOROVOD_GPU_ALLGATHER=MPI HOROVOD_GPU_BROADCAST=MPI pip install -v --upgrade --no-cache-dir horovod==0.13.10 --user

