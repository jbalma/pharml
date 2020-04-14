# Source me

export SCRATCH=/lus/scratch/$USER

if [ $USER == "swowner" ]; then
    umask 002 # all-readable
    INSTALL_BASE=/usr/common/software
else
    INSTALL_BASE=$SCRATCH/condaenv
fi
#export INSTALL_BASE=$SCRATCH/conda
# Configure the installation
export INSTALL_NAME="pytorch-cuda"
export PYTORCH_VERSION="v1.3.0"
export PYTORCH_URL=https://github.com/pytorch/pytorch
export VISION_VERSION="v0.3.0"
export BUILD_DIR=$SCRATCH/pytorch-build/$INSTALL_NAME/$PYTORCH_VERSION
export INSTALL_DIR=$INSTALL_BASE/$INSTALL_NAME/$PYTORCH_VERSION

# Setup programming environment
#module unload PrgEnv-cray
#module load PrgEnv-gnu
#module unload atp
#module unload cray-libsci
#module unload craype-hugepages8M
#module unload craype-broadwell
#module unload gcc
#module load gcc/7.2.0
source /cray/css/users/jbalma/bin/setup_env_cuda10_osprey_V100.sh
source /cray/css/users/jbalma/bin/env_python3.sh
export CRAY_CPU_TARGET=x86-64

# Setup conda
#source /usr/common/software/python/3.6-anaconda-5.2/etc/profile.d/conda.sh
#source /cray/css/users/dctools/anaconda3/etc/profile.d/conda.sh
#conda create $INSTALL_DIR

# Print some stuff
echo "Configuring on $(hostname) as $USER"
echo "  Build directory $BUILD_DIR"
echo "  Install directory $INSTALL_DIR"
