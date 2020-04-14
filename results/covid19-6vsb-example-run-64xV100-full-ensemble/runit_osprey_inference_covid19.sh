#!/bin/bash
#SBATCH -N 1
####SBATCH -C "V100|V10032GB|V10016GB"
#SBATCH -C V10032GB
#SBATCH --mem=0
#SBATCH --ntasks-per-node=8
#SBATCH -p spider
#SBATCH --exclude=spider-0012,spider-0013,spider-0002
####SBATCH -w spider-0013
#SBATCH --exclusive
##SBATCH -e stderr.out
##SBATCH -o stdout.out
#SBATCH --job-name=resnet
#SBATCH -t 48:00:00
####SBATCH -t 1:00:00

#source ./setup_env_cuda10.sh
#unset PYTHONPATH

#source ./setup_env_cuda10_covi19.sh
source ./config_cuda10.sh
unset PYTHONPATH
module rm PrgEnv-cray
#source activate ${INSTALL_DIR}
#MLD_RDK_ENV_INSTALL_DIR=~/cuda10_env
#source activate $MLD_RDK_ENV_INSTALL_DIR
#source activate /home/users/jbalma/cuda10_env_nccl

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
#export HOROVOD_FUSION_THRESHOLD=0
#export HOROVOD_MPI_THREADS_DISABLE=1
#export HOROVOD_FUSION_THRESHOLD=0

d="$(date +%Y)-$(date +%h%m-%s)"

#ENSEMBLE_MODELS=/lus/scratch/jbalma//results_${d}/trained_models
#ENSEMBLE_MODELS=/lus/scratch/jbalma/temp/results_monolithic3/train/monolithic3-mldock-train-bindingdb_2019m4_25-test_bindingdb_2019m4_1of75pct-np-64-lr0.00000001-5,5-layer-32,32x2,2-bs_8-epochs-963-nf-16,16_resumedfrom90/
ENSEMBLE_MODELS=./pretrained-models/mh-gnnx5-ensemble
ENSEMBLE_OUTPUT=./results/covid19_${d}/inference
ENSEMBLE_RAW=./results/covid19_${d}/raw_results
mkdir -p $ENSEMBLE_OUTPUT
mkdir -p $ENSEMBLE_RAW

echo "Starting Ensemble train/test run, saving to:"
echo " -> Model files: ${ENSEMBLE_MODELS}"
echo " -> Inference Output Values: ${ENSEMBLE_OUTPUT}"
echo " -> Raw Run Results: ${ENSEMBLE_RAW}"

list_of_files="6vsb-bindingdb"
#list_of_files="6vsb-fda"

echo "Starting COVID-19 Structure Inference for all structure files (${list_of_files}) for all ensemble members."
echo "========================================================================================"
echo "--> 6LZG: Spike receptor-binding domain complexed with its receptor ACE2: https://www.rcsb.org/structure/6LZG"
echo "--> 6VSB: Prefusion 2019-nCoV spike glycoprotein with a single receptor-binding domain up: https://www.rcsb.org/structure/6vsb"
echo "--> 6LU7: The crystal structure of COVID-19 main protease in complex with an inhibitor N3: https://www.rcsb.org/structure/6LU7"
echo "--> Starting at: $(date)"

#Loop over the PDB IDs of interest
for f in $list_of_files
do
  #Loop over each model of the ensemble
  for n in 0 1 2 3 4
  do
    MAP_TRAIN_NAME="6vsb-fda"
    MAP_TEST_NAME=$f
    
    #MAP_TRAIN_PATH=/lus/scratch/jbalma/avose_backup/data/map/${MAP_TRAIN_NAME}.map
    #MAP_TEST_PATH=/lus/scratch/jbalma/avose_backup/data/map/${MAP_TEST_NAME}.map
    MAP_TRAIN_PATH=/lus/scratch/jbalma/DataSets/Binding/mldock/tools/covid19/data/map/${MAP_TRAIN_NAME}.map
    MAP_TEST_PATH=/lus/scratch/jbalma/DataSets/Binding/mldock/tools/covid19/data/map/${MAP_TEST_NAME}.map

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
    #export UCX_NET_DEVICES="eth0,mlx5_0:1" #,ib0,eth0"   #mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1
    #export DL_COMM_USE_CRCCL=1
    export OMPI_MCA_btl_tcp_if_include=ib0
    #-mca btl_tcp_if_include ens4d1
    NODES=8 #nodes total
    PPN=8 #processer per node
    PPS=4 #processes per socket
    NP=64 #processes total
    NC=4  #job threads per rank
    NT=4  #batching threads per worker
    BS=16 #batch size per rank
    BS_TEST=16 #inference batch size
    #LR0=0.000001 #for BS=2,4,6
    LR0=0.000000001
    MLP_LATENT=32,32
    MLP_LAYERS=2,2
    GNN_LAYERS=5,5
    NUM_FEATURES=16,16
    MODE=classification
    EPOCHS=1

    INFER_OUT="model${n}_${MAP_TEST_NAME}_inference.map"
    
    TEMP_DIR=${SCRATCH}/temp/pharml-covid-${MAP_TEST_NAME}-np-${NP}-lr${LR0}-${GNN_LAYERS}-bs${BS_TEST}
    #rm -rf $TEMP_DIR
    mkdir -p ${TEMP_DIR}
    cp -r /cray/css/users/jbalma/Innovation-Proposals/mldock/mldock-gnn/* ${TEMP_DIR}/
    cd ${TEMP_DIR}
    export SLURM_WORKING_DIR=${TEMP_DIR}


    #Start CUDA MPS Server for Dense GPU nodes
    time srun --cpu_bind=none -p spider -C V100 -l -N ${NODES} --ntasks-per-node=1 -u ./restart_mps.sh 2>&1 |& tee mps_result.txt
    
    #Start the inference run on a single model
    time srun -c ${NC} --hint=multithread --cpu_bind=none -C V100 -p spider -l -N ${NODES} -n ${NP} --ntasks-per-node=${PPN} --ntasks-per-socket=${PPS} -u --cpu_bind=none python mldock_gnn.py \
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
        --inference_only True \
        --hvd True \
        --restore="${ENSEMBLE_MODELS}/model_${n}/checkpoints/model0.ckpt" \
        --inference_out ${INFER_OUT} \
        --epochs 1 2>&1 |& tee log-covid19-${MAP_TEST_NAME}-model-${n}.out

    #create a new directory to store the inference output in
    mkdir -p ${ENSEMBLE_OUTPUT}/model_${n}/${MAP_TEST_NAME}_inference_output
    mkdir -p ${ENSEMBLE_RAW}/model_${n}/${MAP_TEST_NAME}_raw_output
    #copy the raw per-worker inference data to the new directory
    mv ./model${n}_${MAP_TEST_NAME}_inference*.map ${ENSEMBLE_OUTPUT}/model_${n}/${MAP_TEST_NAME}_inference_output/
    #concat the per-worker inference data into a single file
    cat ${ENSEMBLE_OUTPUT}/model_${n}/${MAP_TEST_NAME}_inference_output/model${n}_${MAP_TEST_NAME}_inference*.map > ${ENSEMBLE_OUTPUT}/model_${n}/combined_predictions_${MAP_TEST_NAME}_inference_model${n}.map
    cp ${TEMP_DIR}/log-covid19-${MAP_TEST_NAME}-model-${n}.out ${ENSEMBLE_RAW}/model_${n}/${MAP_TEST_NAME}_raw_output/
    sleep 10
    
    echo "done inferencing ${MAP_TEST_NAME} dataset with ensemble model ${n}."

  done
  echo "Done with ALL ensemble inference tests for ${MAP_TEST_NAME}. Moving on to next PDB Structure"
done
echo "Done testing all compounds against all models."
wait
date


