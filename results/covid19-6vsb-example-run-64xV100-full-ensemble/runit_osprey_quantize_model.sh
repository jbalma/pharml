#!/bin/bash
#SBATCH -N 1
####SBATCH -C "V100|V10032GB|V10016GB"
#SBATCH -C V100
#SBATCH --mem=0
###SBATCH --ntasks-per-node=8
#SBATCH -p spider
#SBATCH --exclude=spider-0012,spider-0013,spider-0002
####SBATCH -w spider-0013
#SBATCH --exclusive
##SBATCH -e stderr.out
##SBATCH -o stdout.out
#SBATCH --job-name=resnet
#SBATCH -t 1:00:00
####SBATCH -t 1:00:00


#source /cray/css/users/jbalma/bin/setup_env_cuda90_osprey_V100.sh
#source /cray/css/users/jbalma/bin/setup_env_cuda10_osprey.sh
#source /cray/css/users/jbalma/bin/env_python3.sh
source ./setup_env_cuda10_quantize.sh
export MLD_RDK_ENV_INSTALL_DIR=~/cuda10_env
source activate $MLD_RDK_ENV_INSTALL_DIR

which python

export SCRATCH=/lus/scratch/jbalma
#export CRAY_CUDA_MPS=1
export CUDA_VISIBLE_DEVICES=0
#export CRAY_CUDA_PROXY=1

echo "Running..."

#export CRAY_CUDA_MPS=1
#export TF_FP16_CONV_USE_FP32_COMPUTE=0
#export TF_FP16_MATMUL_USE_FP32_COMPUTE=0
#export HOROVOD_TIMELINE=${SCRATCH_PAD}/timeline.json
#export HOROVOD_FUSION_THRESHOLD=256000000
#export HOROVOD_FUSION_THRESHOLD=500000
#export HOROVOD_FUSION_THRESHOLD=0
export HOROVOD_MPI_THREADS_DISABLE=1
export HOROVOD_FUSION_THRESHOLD=0

d="$(date +%Y)-$(date +%h%m-%s)"
save_dir_name=pharml_results_quantized_${d}
ORIGINAL_MODELS_DIR=/lus/scratch/jbalma/pharml_results_5layer/ensemble_models

ENSEMBLE_MODELS=/lus/scratch/jbalma/$save_dir_name/ensemble_models
ENSEMBLE_OUTPUT=/lus/scratch/jbalma/$save_dir_name/ensemble_outputs
ENSEMBLE_RAW=/lus/scratch/jbalma/$save_dir_name/ensemble_raw_runs

echo "Starting Ensemble train/test run, saving to:"
echo " -> Model files: ${ENSEMBLE_MODELS}"
echo " -> Inference Output Values: ${ENSEMBLE_OUTPUT}"
echo " -> Raw Run Results: ${ENSEMBLE_RAW}"


#list_of_files="bindingdb_2019m4_10of25pct
#               bindingdb_2019m4_15of25pct
#list_of_files="bindingdb_2019m4_20of25pct"
list_of_models="/lus/scratch/jbalma/pharml_results_5layer/ensemble_models/model_0"
#                /lus/scratch/jbalma/pharml_results_5layer/ensemble_models/model_1
#                /lus/scratch/jbalma/pharml_results_5layer/ensemble_models/model_2
#                /lus/scratch/jbalma/pharml_results_5layer/ensemble_models/model_3
#                /lus/scratch/jbalma/pharml_results_5layer/ensemble_models/model_4"

model_n=0
for f in $list_of_models
do
    MODEL_PATH_TO_QUANTIZE=$f
    MAP_TRAIN_NAME=l0_1pct_train
    MAP_TEST_NAME=bindingdb_2019m4_1of75pct
    MAP_TEST_BIG_NAME=bindingdb_2019m4_75
    MAP_TEST_ZINC_NAME=4ib4_zinc15

    MAP_TRAIN_PATH=/lus/scratch/avose/data/map/${MAP_TRAIN_NAME}.map
    MAP_TEST_PATH=/lus/scratch/avose/data/map/${MAP_TEST_NAME}.map
    MAP_TEST_BIG_PATH=/lus/scratch/avose/data/map/${MAP_TEST_BIG_NAME}.map
    MAP_TEST_ZINC_PATH=/lus/scratch/avose/data_zinc15/map/${MAP_TEST_ZINC_NAME}.map

    echo "Model number = ${model_n}"
    echo "Quantizing model from: ${MODEL_PATH_TO_QUANTIZE}"
    echo "Running with input data ${MAP_TRAIN_PATH}"
    echo "test data from: ${MAP_TEST_PATH}"
    echo "big test data from: ${MAP_TEST_BIG_PATH}"
    echo "zinc test data from: ${MAP_TEST_ZINC_PATH}"

#-rw-r--r--. 1 avose criemp 9.3M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_a.map
#-rw-r--r--. 1 avose criemp 8.6M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_b.map
#-rw-r--r--. 1 avose criemp 9.0M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_c.map
#-rw-r--r--. 1 avose criemp 8.9M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_d.map
#-rw-r--r--. 1 avose criemp 9.6M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_e.map
#-rw-r--r--. 1 avose criemp 149M Jun 23 07:06 /lus/scratch/avose/data/map/bindingdb_2019m4_75.map


    NODES=1 #nodes total
    PPN=1 #processer per node
    PPS=1 #processes per socket
    NP=1 #processes total
    NC=36  #job threads per rank
    NT=2  #batching threads per worker
    BS=16 #batch size per rank
    BS_TEST=32 #inference batch size
    #LR0=0.000001 #for BS=2,4,6
    LR0=0.00000001
    MLP_LATENT=32,32
    MLP_LAYERS=2,2
    GNN_LAYERS=5,5
    NUM_FEATURES=16,16
    MODE=classification
    EPOCHS=1000

    TEMP_DIR=${SCRATCH}/temp/quantized_check_model${model_n}-test_${MAP_TEST_NAME}-np-${NP}-lr${LR0}-${GNN_LAYERS}-layer-${MLP_LATENT}x${MLP_LAYERS}-bs_${BS}-epochs-${EPOCHS}-nf-${NUM_FEATURES}
    rm -rf $TEMP_DIR
    mkdir -p ${TEMP_DIR}
    cp -r -v /cray/css/users/jbalma/Innovation-Proposals/mldock/mldock-gnn/* ${TEMP_DIR}/
    cd ${TEMP_DIR}
    export SLURM_WORKING_DIR=${TEMP_DIR}

    echo
    pwd
    ls
    echo

    echo "Running Train..."
    date
    srun -C V100 -p spider -c ${NC} --hint=multithread -l -N ${NODES} -n ${NP} -u --exclusive --cpu_bind=rank_ldom --accel-bind=g,v python mldock_gnn_quantize.py \
        --map_train ${MAP_TRAIN_PATH} \
        --map_test ${MAP_TEST_PATH} \
        --batch_size ${BS} \
        --batch_size_test ${BS_TEST} \
        --mlp_latent ${MLP_LATENT} \
        --mlp_layers ${MLP_LAYERS} \
        --gnn_layers ${GNN_LAYERS} \
        --hvd True \
        --lr_init ${LR0} \
        --use_clr True \
        --plot_history True \
        --num_features ${NUM_FEATURES} \
        --data_threads ${NT} \
        --mode ${MODE} \
        --inference_only True \
        --restore="${MODEL_PATH_TO_QUANTIZE}/checkpoints/model0.ckpt" \
        --epochs ${EPOCHS} 2>&1 |& tee trainrun.out


    let "model_n=model_n+1"

done
echo "Done training/testing all monolithic-trained models"
wait
date


