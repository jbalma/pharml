#!/bin/bash
#SBATCH -N 4
#SBATCH -C P100
#SBATCH --exclusive
#SBATCH -t 1:00:00
#SBATCH -o tiger_mldock.out


ulimit -s unlimited
source /cray/css/users/jbalma/bin/setup_env_cuda10.sh 
source /cray/css/users/jbalma/bin/env_python3.sh
source activate ~/cuda10_env
which python

export SCRATCH=/lus/scratch/jbalma
export MPICH_ENV_DISPLAY=1
export MPICH_VERSION_DISPLAY=1
export MPICH_CPUMASK_DISPLAY=1
#export MPICH_COLL_SYNC=1 #force everyone into barrier before each collective
export MPICH_RDMA_ENABLED_CUDA=1
export MPICH_MAX_THREAD_SAFETY=multiple
#export CRAY_CUDA_MPS=1
export CUDA_VISIBLE_DEVICES=0
#export CRAY_CUDA_PROXY=1

echo "Running..."

#export TF_FP16_CONV_USE_FP32_COMPUTE=0
#export TF_FP16_MATMUL_USE_FP32_COMPUTE=0
#export HOROVOD_TIMELINE=${SCRATCH_PAD}/timeline.json
#HYPNO="/cray/css/users/kjt/bin/hypno --plot node_power"
#export HOROVOD_FUSION_THRESHOLD=256000000
#export HOROVOD_MPI_THREADS_DISABLE=1
export HOROVOD_FUSION_THRESHOLD=0
PROF_LINE="-m cProfile -o pyprof_mldock.out"

#MAP_TRAIN_NAME=train_small
#MAP_TEST_NAME=test_small
#MAP_TRAIN_PATH=/lus/scratch/jbalma/DataSets/mldock-full-jbalma/mldock/mlvoxelizer/sdf_splitter/data/map/${MAP_TRAIN_NAME}.map
#MAP_TEST_PATH=/lus/scratch/jbalma/DataSets/mldock-full-jbalma/mldock/mlvoxelizer/sdf_splitter/data/map/${MAP_TEST_NAME}.map

MAP_TRAIN_NAME=pdb_3bcu_train
MAP_TEST_NAME=pdb_3bcu_test
MAP_TRAIN_PATH=/lus/scratch/jbalma/DataSets/data/map/${MAP_TRAIN_NAME}.map
MAP_TEST_PATH=/lus/scratch/jbalma/DataSets/data/map/${MAP_TEST_NAME}.map

NODES=8
NP=8
BS=4
PPN=1 #processer per node
PPS=1 #processes per socket
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


TEMP_DIR=${SCRATCH}/temp/mldock-${MAP_TRAIN_NAME}-${MAP_TEST_NAME}-np_${NP}-lr_${LR0}-${GNN_LAYERS}_layer-${MLP_LATENT}x${MLP_LAYERS}x${MLP_FILTERS}-bs_${BS}-epochs_${EPOCHS}-syn_${SYN_MODE}
rm -rf $TEMP_DIR
mkdir -p ${TEMP_DIR}
cp -r ./* ${TEMP_DIR}/
cd ${TEMP_DIR}
export SLURM_WORKING_DIR=${TEMP_DIR}

echo "Running Train..."
date
time srun --pty -C P100 -u -N ${NODES} -n ${NP} --exclusive --cpu_bind=rank_ldom -u python mldock_gnn.py \
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

#time srun --pty -C P100 -u -N ${NODES} -n ${NP} --exclusive --cpu_bind=rank_ldom python mldock_gnn.py \
#    --map_train ${MAP_TRAIN_PATH} \
#    --map_test ${MAP_TEST_PATH} \
#    --batch_size ${BS} \
#    --mlp_latent ${MLP_LATENT} \
#    --mlp_layers ${MLP_LAYERS} \
#    --mlp_filters ${MLP_FILTERS} \
#    --num_proc_steps ${GNN_LAYERS} \
#    --lr_init ${LR0} \
#    --use_clr True \
#    --hvd True \
#    --synthetic ${SYN_MODE} \
#    --epochs ${EPOCHS} 2>&1 | tee train_osprey.log
date

