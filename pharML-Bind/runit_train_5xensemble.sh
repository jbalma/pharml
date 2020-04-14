#!/bin/bash
#SBATCH -N 4
####SBATCH -C "V100|V10032GB|V10016GB"
#SBATCH -C V100
####SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH -p spider
#SBATCH --exclude=spider-0012,spider-0013,spider-0002
#SBATCH --exclusive
#SBATCH --job-name=pharml-bind
#SBATCH -t 24:00:00

source ./config_cuda10.sh
unset PYTHONPATH
module rm PrgEnv-cray

INSTALL_DIR=/lus/scratch/jbalma/condenv-cuda10-pharml

#conda create -y --prefix $INSTALL_DIR python=3.6 cudatoolkit=10.0 cudnn
source activate $INSTALL_DIR/
export PATH=${INSTALL_DIR}/bin:${PATH} #/home/users/${USER}/.local/bin:${PATH}
#conda install -y -c conda-forge rdkit biopython scipy dask
#conda install -y -c anaconda pip
#python -m pip install --force-reinstall graph_nets "tensorflow_gpu>=1.15,<2" "dm-sonnet<2" "tensorflow_probability<0.9"
echo $CUDATOOLKIT_HOME
which mpicc
which mpic++
which gcc
which python

#conda install cmake
#conda install pip
#pip uninstall horovod
export CMAKE_CXX_COMPILER=$MPI_CXX
export CMAKE_CC_COMPILER=$MPI_CC
export HOROVOD_ALLOW_MIXED_GPU_IMPL=0

#HOROVOD_BUILD_ARCH_FLAGS="-mavx256" HOROVOD_CUDA_HOME=${CUDATOOLKIT_HOME} HOROVOD_MPICXX_SHOW="mpic++ -show" HOROVOD_HIERARCHICAL_ALLREDUCE=0 HOROVOD_MPI_HOME=${MPI_PATH} HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip install --global-option=build_ext --global-option="-I ${CUDATOOLKIT_HOME}/include" --no-cache-dir horovod

HOROVOD_BUILD_ARCH_FLAGS="-mavx256" HOROVOD_WITH_TENSORFLOW=1 HOROVOD_WITHOUT_PYTORCH=1 HOROVOD_WITHOUT_MXNET=1 pip install --no-cache-dir horovod
#exit
conda list

export SCRATCH=/lus/scratch/jbalma
export CRAY_CUDA_MPS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export TF_ENABLE_AUTO_MIXED_PRECISION=1
#export CRAY_CUDA_PROXY=1
echo "Running..."
#export CRAY_CUDA_MPS=1
export TF_FP16_CONV_USE_FP32_COMPUTE=0
export TF_FP16_MATMUL_USE_FP32_COMPUTE=0
#export HOROVOD_TIMELINE=${SCRATCH_PAD}/timeline.json
#export HOROVOD_FUSION_THRESHOLD=256000000
#export HOROVOD_FUSION_THRESHOLD=500000
export HOROVOD_FUSION_THRESHOLD=0
#export HOROVOD_MPI_THREADS_DISABLE=1
#export HOROVOD_FUSION_THRESHOLD=0


d="$(date +%Y)-$(date +%h%m-%s)"

#ENSEMBLE_MODELS=./pretrained-models/mh-gnnx5-ensemble
#ENSEMBLE_OUTPUT=./results/covid19_6vsb/inference
ENSEMBLE_RAW=./results/covid19_6vsb/raw_results
mkdir -p $ENSEMBLE_OUTPUT
mkdir -p $ENSEMBLE_RAW

echo "Starting Ensemble train/test run, saving to:"
echo " -> Model files: ${ENSEMBLE_MODELS}"
echo " -> Inference Output Values: ${ENSEMBLE_OUTPUT}"
echo " -> Raw Run Results: ${ENSEMBLE_RAW}"

##list_of_files="6vsb-full-bindingdb-fda"
#list_of_files="6vsb-fda"
#list_of_files="6vsb-bindingdb"
list_of_files="bindingdb_2019m4_5of31of75_a bindingdb_2019m4_5of31of75_b bindingdb_2019m4_5of31of75_c bindingdb_2019m4_5of31of75_d bindingdb_2019m4_5of31of75_e"  

#/lus/scratch/jbalma/DataSets/Binding/bindingdb_2019m4/data/map/bindingdb_2019m4_10of69of75_a.map

echo "Starting Ensemble Training Run. training on (${list_of_files}) for all ensemble members."
echo "========================================================================================"

echo "--> Starting at: $(date)"

#Loop over the PDB IDs of interest
n=0
for f in $list_of_files
do
    MAP_TRAIN_NAME=$f
    #MAP_TEST_NAME=bindingdb_2019m4_10of69of75_a
    MAP_TEST_NAME=bindingdb_2019m4_1of69of75_a
    MAP_TRAIN_PATH=/lus/scratch/jbalma/DataSets/Binding/bindingdb_2019m4/data/map/${MAP_TRAIN_NAME}.map
    MAP_TEST_PATH=/lus/scratch/jbalma/DataSets/Binding/bindingdb_2019m4/data/map/${MAP_TEST_NAME}.map
    #MAP_TRAIN_PATH=/lus/scratch/jbalma/avose_backup/data/map/${MAP_TRAIN_NAME}.map
    #MAP_TRAIN_PATH=/lus/scratch/jbalma/DataSets/Binding/mldock/tools/covid19/data/map/
    
    #MAP_TEST_PATH=/lus/scratch/jbalma/avose_backup/data/map/${MAP_TEST_NAME}.map
    #MAP_TRAIN_PATH=/lus/scratch/jbalma/DataSets/Binding/mldock/tools/covid19/data/map/${MAP_TRAIN_NAME}.map
    #MAP_TEST_PATH=/lus/scratch/jbalma/DataSets/Binding/mldock/tools/covid19/data/map/${MAP_TEST_NAME}.map
    #MAP_TEST_PATH=/lus/scratch/jbalma/DataSets/Binding/mldock/tools/covid19-fda-bindingdb/data-6vsb-bindingdb-fda-all/map/${MAP_TEST_NAME}.map
    
    #MAP_TEST_PATH=/lus/scratch/jbalma/DataSets/Binding/mldock/tools/covid19-fda-bindingdb/data-6vsb-bindingdb/map/${MAP_TEST_NAME}.map
    #MAP_TEST_PATH=/lus/scratch/jbalma/DataSets/Binding/mldock/tools/covid19-fda-bindingdb/data-6vsb-bindingdb-full/map/${MAP_TEST_NAME}.map
    #MAP_TEST_PATH=/lus/scratch/jbalma/DataSets/Binding/mldock/tools/covid19-fda-bindingdb/data-6vsb-bindingdb-fda/map/${MAP_TEST_NAME}.map
    echo "Model number = ${n}"
    echo "Running with training input data ${MAP_TRAIN_PATH}"
    echo "test data from: ${MAP_TEST_PATH}"


    export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
    #export OMPI_MCA_btl_openib_allow_ib=false
    #export OMPI_MCA_btl_openib_allow_ib=true
    #export OMPI_MCA_btl=^openib
    #export UCX_TLS="cma,dc_mlx5,posix,rc,rc_mlx5,self,sm,sysv,tcp,ud,ud_mlx5"
    #export UCX_MEMTYPE_CACHE=n
    #export UCX_ACC_DEVICES=""
    #export UCX_NET_DEVICES="ib0,eth0,mlx5_0:1" #,ib0,eth0"   #mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
    #export DL_COMM_USE_CRCCL=1
    export OMPI_MCA_btl_tcp_if_include=ib0
    #-mca btl_tcp_if_include ens4d1
    NODES=4 #nodes total
    PPN=8 #processer per node
    PPS=4 #processes per socket
    NP=32 #processes total
    NC=9  #job threads per rank
    NT=4  #batching threads per worker
    BS=8 #batch size per rank
    BS_TEST=8 #inference batch size
    #LR0=0.000001 #for BS=2,4,6
    LR0=0.000000001
    MLP_LATENT=32,32
    MLP_LAYERS=2,2
    GNN_LAYERS=5,5
    NUM_FEATURES=16,16
    MODE=classification
    EPOCHS=1000

    #INFER_OUT="model${n}_${MAP_TEST_NAME}_inference.map"
    
    TEMP_DIR=${SCRATCH}/temp/pharml-bind-new5xensemble-train${MAP_TRAIN_NAME}_test${MAP_TEST_NAME}-np-${NP}-lr${LR0}-bs${BS}-model${n}
    rm -rf $TEMP_DIR
    mkdir -p ${TEMP_DIR}
    cp -r /cray/css/users/jbalma/Innovation-Proposals/mldock/mldock-gnn/* ${TEMP_DIR}/
    cd ${TEMP_DIR}
    export SLURM_WORKING_DIR=${TEMP_DIR}


    #Start CUDA MPS Server for Dense GPU nodes
    time srun --cpu_bind=none -p spider -C V100 -l -N ${NODES} --ntasks-per-node=1 -n ${NODES} -u ./restart_mps.sh 2>&1 |& tee mps_result.txt

    #Wait a few seconds before starting the main job
    sleep 10    

    #Start the inference run on a single model
    time srun -c ${NC} --hint=multithread -C V100 -p spider -l -N ${NODES} -n ${NP} --ntasks-per-node=${PPN} --ntasks-per-socket=${PPS} -u --cpu_bind=rank_ldom --accel-bind=g,v python mldock_gnn.py \
        --map_train ${MAP_TRAIN_PATH} \
        --map_test ${MAP_TEST_PATH} \
        --batch_size ${BS} \
        --batch_size_test ${BS_TEST} \
        --mlp_latent ${MLP_LATENT} \
        --mlp_layers ${MLP_LAYERS} \
        --gnn_layers ${GNN_LAYERS} \
        --num_features ${NUM_FEATURES} \
        --data_threads ${NT} \
        --mode ${MODE} \
        --hvd True \
        --lr_init ${LR0} \
        --use_clr True \
        --plot_history True \
        --epochs ${EPOCHS} 2>&1 |& tee log-train-${MAP_TEST_NAME}-model-${n}.out

    cp -v ${TEMP_DIR}/log-${MAP_TEST_NAME}-model-${n}.out ${ENSEMBLE_RAW}/model_${n}/${MAP_TEST_NAME}/

    let "n=n+1"
    sleep 10
    

  done
echo "Done with ALL ensemble training ${MAP_TRAIN_NAME}. and test set ${MAP_TEST_NAME} "
wait
date


