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


#source /cray/css/users/jbalma/bin/setup_env_cuda90_osprey_V100.sh
unset PYTHONPATH
#source /cray/css/users/jbalma/bin/env_python3.sh
source /cray/css/users/jbalma/bin/setup_env_cuda10_osprey_V100.sh
source /cray/css/users/jbalma/bin/env_python3.sh

export MLD_RDK_ENV_INSTALL_DIR=/home/users/jbalma/cuda10_env_pharml
source activate $MLD_RDK_ENV_INSTALL_DIR
#conda install -y -c anaconda pip
#pip install tensorflow_gpu==1.15 graph_nets sonnet tensorflow_probability wrapt networkx
pip install networkx --upgrade
pip install timeit
#pip install numpy pyparsing
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

#list_of_files="bindingdb_2019m4_10of25pct"
list_of_files="l0_1pct_train"
model_n=0
#for i in $(seq 1 $END); do echo $i; done
for f in $list_of_files
do
    MAP_TRAIN_NAME=$f
    MAP_TEST_NAME="l0_1pct_test"

    MAP_TRAIN_PATH=/lus/scratch/jbalma/avose_backup/data/map/${MAP_TRAIN_NAME}.map
    #MAP_TRAIN_PATH=/lus/scratch/jbalma/data/mldock/tools/${MAP_TRAIN_NAME}.map
    MAP_TEST_PATH=/lus/scratch/jbalma/avose_backup/data/map/${MAP_TEST_NAME}.map
    #MAP_TEST_PATH=/lus/scratch/jbalma/data/mldock/tools/${MAP_TEST_NAME}.map

    echo "Running with input data ${MAP_TRAIN_PATH}"
    echo "test data from: ${MAP_TEST_PATH}"

#-rw-r--r--. 1 avose criemp 9.3M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_a.map
#-rw-r--r--. 1 avose criemp 8.6M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_b.map
#-rw-r--r--. 1 avose criemp 9.0M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_c.map
#-rw-r--r--. 1 avose criemp 8.9M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_d.map
#-rw-r--r--. 1 avose criemp 9.6M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_e.map
#-rw-r--r--. 1 avose criemp 149M Jun 23 07:06 /lus/scratch/avose/data/map/bindingdb_2019m4_75.map


    NODES=1 #nodes total
    PPN=1 #processer per node
    PPS=4 #processes per socket
    NP=1 #processes total
    NC=9  #job threads per rank
    NT=2  #batching threads per worker
    BS=5 #batch size per rank
    BS_TEST=5 #inference batch size
    #LR0=0.000001 #for BS=2,4,6
    LR0=0.000000001
    MLP_LATENT=32,32
    MLP_LAYERS=2,2
    GNN_LAYERS=5,5
    NUM_FEATURES=16,16
    MODE=classification
    EPOCHS=100

    #if [ $model_n -gt 2 ]
    #then
    #    let "BS=BS/2"
    #fi
    

    TEMP_DIR=${SCRATCH}/temp/latest-pharmlactive-train-${MAP_TRAIN_NAME}-test_${MAP_TEST_NAME}-np-${NP}-lr${LR0}-${GNN_LAYERS}-layer-${MLP_LATENT}x${MLP_LAYERS}-bs_${BS}-epochs-${EPOCHS}-nf-${NUM_FEATURES}_fresh
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
    time srun -x spider-0002 -c ${NC} --hint=multithread -C V100 -p spider -l -N ${NODES} -n ${NP} --ntasks-per-node=${PPN} --ntasks-per-socket=${PPS} -u --exclusive --cpu_bind=rank_ldom --accel-bind=g,v python mldock_gnn.py \
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
        --epochs ${EPOCHS} 2>&1 |& tee trainrun.out

done
echo "Done training"
wait
date


