#!/bin/bash
#SBATCH -N 4
####SBATCH -C "V100|V10032GB|V10016GB"
#SBATCH -C V100
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


source /cray/css/users/jbalma/bin/setup_env_cuda90_osprey_V100.sh
source /cray/css/users/jbalma/bin/env_python3.sh
export MLD_RDK_ENV_INSTALL_DIR=~/cuda90_env
source activate $MLD_RDK_ENV_INSTALL_DIR

which python

export SCRATCH=/lus/scratch/jbalma
#export CRAY_CUDA_MPS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
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
#export HOROVOD_FUSION_THRESHOLD=0

d="$(date +%Y)-$(date +%h%m-%s)"

ENSEMBLE_MODELS=/lus/scratch/jbalma/pharml_results_monolithic_${d}/ensemble_models
ENSEMBLE_OUTPUT=/lus/scratch/jbalma/pharml_results_monolithic_${d}/ensemble_outputs
ENSEMBLE_RAW=/lus/scratch/jbalma/pharml_results_monolithic_${d}/ensemble_raw_runs

echo "Starting Ensemble train/test run, saving to:"
echo " -> Model files: ${ENSEMBLE_MODELS}"
echo " -> Inference Output Values: ${ENSEMBLE_OUTPUT}"
echo " -> Raw Run Results: ${ENSEMBLE_RAW}"


#list_of_files="bindingdb_2019m4_10of25pct
#               bindingdb_2019m4_15of25pct
list_of_files="bindingdb_2019m4_20of25pct"
#list_of_files="bindingdb_2019m4_25"
model_n=2
for f in $list_of_files
do
    MAP_TRAIN_NAME=$f
    MAP_TEST_NAME=bindingdb_2019m4_1of75pct
    MAP_TEST_BIG_NAME=bindingdb_2019m4_75
    MAP_TEST_ZINC_NAME=4ib4_zinc15

    MAP_TRAIN_PATH=/lus/scratch/avose/data/map/${MAP_TRAIN_NAME}.map
    #MAP_TRAIN_PATH=/lus/scratch/jbalma/data/mldock/tools/${MAP_TRAIN_NAME}.map
    MAP_TEST_PATH=/lus/scratch/avose/data/map/${MAP_TEST_NAME}.map
    #MAP_TEST_PATH=/lus/scratch/jbalma/data/mldock/tools/${MAP_TEST_NAME}.map
    MAP_TEST_BIG_PATH=/lus/scratch/avose/data/map/${MAP_TEST_BIG_NAME}.map
    MAP_TEST_ZINC_PATH=/lus/scratch/avose/data_zinc15/map/${MAP_TEST_ZINC_NAME}.map

    echo "Model number = ${model_n}"
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


    NODES=4 #nodes total
    PPN=8 #processer per node
    PPS=4 #processes per socket
    NP=32 #processes total
    NC=9  #job threads per rank
    NT=2  #batching threads per worker
    BS=16 #batch size per rank
    BS_TEST=32 #inference batch size
    #LR0=0.000001 #for BS=2,4,6
    LR0=0.000000001
    MLP_LATENT=32,32
    MLP_LAYERS=2,2
    GNN_LAYERS=5,5
    NUM_FEATURES=16,16
    MODE=classification
    EPOCHS=1

    #if [ $model_n -gt 2 ]
    #then
    #    let "BS=BS/2"
    #fi
    

    TEMP_DIR=${SCRATCH}/temp/monolithic${model_n}-mldock-train-${MAP_TRAIN_NAME}-test_${MAP_TEST_NAME}-np-${NP}-lr${LR0}-${GNN_LAYERS}-layer-${MLP_LATENT}x${MLP_LAYERS}-bs_${BS}-epochs-${EPOCHS}-nf-${NUM_FEATURES}_resumedfrom227_final
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
    time srun -c ${NC} --hint=multithread -l -N ${NODES} -n ${NP} --ntasks-per-node=${PPN} --ntasks-per-socket=${PPS} -u --exclusive --cpu_bind=rank_ldom --accel-bind=g,v python mldock_gnn.py \
        --map_train ${MAP_TRAIN_PATH} \
        --map_test ${MAP_TEST_PATH} \
        --batch_size ${BS} \
        --batch_size_test ${BS_TEST} \
        --mlp_latent ${MLP_LATENT} \
        --mlp_layers ${MLP_LAYERS} \
        --gnn_layers ${GNN_LAYERS} \
        --lr_init ${LR0} \
        --use_clr True \
        --hvd True \
        --plot_history True \
        --num_features ${NUM_FEATURES} \
        --data_threads ${NT} \
        --mode ${MODE} \
        --restore="/lus/scratch/jbalma/temp/results_monolithic2/train/monolithic2-mldock-train-227epochs/checkpoints/model0.ckpt" \
        --epochs ${EPOCHS} 2>&1 |& tee trainrun.out

    #--restore="/lus/scratch/jbalma/temp/monolithic2-mldock-train-180epochs/checkpoints/model0.ckpt"
    mkdir -p ${ENSEMBLE_MODELS}/model_${model_n}/checkpoints
    mkdir -p ${ENSEMBLE_RAW}/model_${model_n}
    cp -v -r checkpoints/* ${ENSEMBLE_MODELS}/model_${model_n}/checkpoints/
    cp -v -r ${TEMP_DIR} ${ENSEMBLE_RAW}/model_${model_n}/

    echo "done training model $model_n"
    echo "saved model to ${ENSEMBLE_MODELS}/model_${model_n}/checkpoints..."
    echo "saved raw train run data to ${ENSEMBLE_RAW}/model_${model_n}..."

    sleep 10

    INFER_OUT=inference.map

    TEMP_DIR=${SCRATCH}/temp/monolithic${model_n}-mldock-infer${MAP_TEST_BIG_NAME}-np-${NP}-lr${LR0}-${GNN_LAYERS}-layer-${MLP_LATENT}x${MLP_LAYERS}-bs_${BS_TEST}-nf-${NUM_FEATURES}
    rm -rf $TEMP_DIR
    mkdir -p ${TEMP_DIR}
    cp -r -v /cray/css/users/jbalma/Innovation-Proposals/mldock/mldock-gnn/* ${TEMP_DIR}/
    cd ${TEMP_DIR}
    export SLURM_WORKING_DIR=${TEMP_DIR}

    echo
    pwd
    ls
    echo

    echo "Running 75% test for $model_n..."
    date

    time srun -c ${NC} --hint=multithread -l -N ${NODES} -n ${NP} --ntasks-per-node=${PPN} --ntasks-per-socket=${PPS} -u --exclusive --cpu_bind=rank_ldom --accel-bind=g,v python mldock_gnn.py \
        --map_train ${MAP_TRAIN_PATH} \
        --map_test ${MAP_TEST_BIG_PATH} \
        --batch_size ${BS} \
        --batch_size_test ${BS_TEST} \
        --mlp_latent ${MLP_LATENT} \
        --mlp_layers ${MLP_LAYERS} \
        --gnn_layers ${GNN_LAYERS} \
        --lr_init ${LR0} \
        --use_clr True \
        --hvd True \
        --num_features ${NUM_FEATURES} \
        --data_threads ${NT} \
        --mode ${MODE} \
        --inference_only True \
        --restore="${ENSEMBLE_MODELS}/model_${model_n}/checkpoints/model0.ckpt" \
        --inference_out ${INFER_OUT} \
        --epochs 1 2>&1 |& tee testrun.out

    mkdir -p ${ENSEMBLE_OUTPUT}/model_${model_n}/inference_output
    cp ./inference*.map ${ENSEMBLE_OUTPUT}/model_${model_n}/inference_output/
    cat ./inference*.map > ${ENSEMBLE_OUTPUT}/model_${model_n}/inference_model${model_n}.map
    cp -v -r ${TEMP_DIR} ${ENSEMBLE_RAW}/model_${model_n}/
    sleep 10

    echo "done with 75% dataset test using $model_n"
    echo "saved 75% inference output to ${ENSEMBLE_OUTPUT}/model_${model_n}/inference_model${model_n}.out..."
    echo "saved 75% raw run data to ${ENSEMBLE_RAW}/model_${model_n}/"


    INFER_OUT=zinc_inference.map

    TEMP_DIR=${SCRATCH}/temp/monolithic${model_n}-mldock-zinctest${MAP_TEST_ZINC_NAME}-np-${NP}-lr${LR0}-${GNN_LAYERS}-layer-${MLP_LATENT}x${MLP_LAYERS}-bs_${BS_TEST}-nf-${NUM_FEATURES}
    rm -rf $TEMP_DIR
    mkdir -p ${TEMP_DIR}
    cp -r -v /cray/css/users/jbalma/Innovation-Proposals/mldock/mldock-gnn/* ${TEMP_DIR}/
    cd ${TEMP_DIR}
    export SLURM_WORKING_DIR=${TEMP_DIR}

    echo
    pwd
    ls
    echo

    echo "Starting 4IB4 ZINC test for $model_n..."
    date

    time srun -c ${NC} --hint=multithread -l -N ${NODES} -n ${NP} --ntasks-per-node=${PPN} --ntasks-per-socket=${PPS} -u --exclusive --cpu_bind=rank_ldom --accel-bind=g,v python mldock_gnn.py \
        --map_train ${MAP_TRAIN_PATH} \
        --map_test ${MAP_TEST_ZINC_PATH} \
        --batch_size ${BS} \
        --batch_size_test ${BS_TEST} \
        --mlp_latent ${MLP_LATENT} \
        --mlp_layers ${MLP_LAYERS} \
        --gnn_layers ${GNN_LAYERS} \
        --lr_init ${LR0} \
        --use_clr True \
        --hvd True \
        --num_features ${NUM_FEATURES} \
        --data_threads ${NT} \
        --mode ${MODE} \
        --inference_only True \
        --restore="${ENSEMBLE_MODELS}/model_${model_n}/checkpoints/model0.ckpt" \
        --inference_out ${INFER_OUT} \
        --epochs 1 2>&1 |& tee zinctest.out

    mkdir -p ${ENSEMBLE_OUTPUT}/model_${model_n}/zinc_inference_output
    cp ./zinc_inference*.map ${ENSEMBLE_OUTPUT}/model_${model_n}/zinc_inference_output/
    cat ./zinc_inference*.map > ${ENSEMBLE_OUTPUT}/model_${model_n}/zinc_inference_model${model_n}.map
    cp -v -r ${TEMP_DIR} ${ENSEMBLE_RAW}/model_${model_n}/
    sleep 10
    
    echo "done with zinc dataset test using $model_n"
    echo "saved zinc inference output to ${ENSEMBLE_OUTPUT}/model_${model_n}/zinc_inference_model${model_n}.map..."
    echo "saved zinc raw run data to ${ENSEMBLE_RAW}/model_${model_n}..."

    let "model_n=model_n+1"

done
echo "Done training/testing all monolithic-trained models"
wait
date


