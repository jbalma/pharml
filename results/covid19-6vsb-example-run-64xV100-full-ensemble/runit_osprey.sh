#!/bin/bash
#SBATCH -N 4
###SBATCH -C "V100|V10032GB|V10016GB"
#SBATCH -C V100
#SBATCH -p spider
#SBATCH --exclusive
#SBATCH -t 1:00:00


#ulimit -s unlimited
#source /cray/css/users/jbalma/bin/setup_env_cuda10_osprey_V100.sh  #setup_env_cuda90_osprey_V100.sh
#source /cray/css/users/jbalma/bin/env_python3.sh
source /cray/css/users/jbalma/bin/setup_env_cuda90_osprey_V100.sh
source /cray/css/users/jbalma/bin/env_python.sh
which python

export SCRATCH=/lus/scratch/jbalma
#export CRAY_CUDA_MPS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
#export CUDA_VISIBLE_DEVICES=${SLURM_LOCALID}
#export CUDA_VISIBLE_DEVICES=0
#unset CUDA_VISIBLE_DEVICES
#export CRAY_CUDA_PROXY=1

echo "Running..."

#export CRAY_CUDA_MPS=1
#export TF_FP16_CONV_USE_FP32_COMPUTE=0
#export TF_FP16_MATMUL_USE_FP32_COMPUTE=0
#export HOROVOD_TIMELINE=${SCRATCH_PAD}/timeline.json
#HYPNO="/cray/css/users/kjt/bin/hypno --plot node_power"
#export HOROVOD_FUSION_THRESHOLD=256000000
#export HOROVOD_MPI_THREADS_DISABLE=1
#export HOROVOD_FUSION_THRESHOLD=0
PROF_LINE="-m cProfile -o pyprof_mldock.out"

#MAP_TRAIN_NAME=train_med
#MAP_TEST_NAME=test_med
#MAP_TRAIN_PATH=/lus/scratch/jbalma/DataSets/mldock-full-jbalma/mldock/mlvoxelizer/sdf_splitter/data/map/${MAP_TRAIN_NAME}.map
#MAP_TEST_PATH=/lus/scratch/jbalma/DataSets/mldock-full-jbalma/mldock/mlvoxelizer/sdf_splitter/data/map/${MAP_TEST_NAME}.map

#MAP_TRAIN_NAME=train_E
#MAP_TEST_NAME=test_E
#MAP_TRAIN_NAME=A_ca2_1bnm
#MAP_TEST_NAME=A_ca4_3f7b
#MAP_TRAIN_NAME=train_D
#MAP_TEST_NAME=test_D
#MAP_TRAIN_NAME=train_B
#MAP_TEST_NAME=test_B
#MAP_TRAIN_PATH=/lus/scratch/jbalma/DataSets/mldock/mlvoxelizer/dataset-B-graph/map/${MAP_TRAIN_NAME}.map
#MAP_TEST_PATH=/lus/scratch/jbalma/DataSets/mldock/mlvoxelizer/dataset-B-graph/map/${MAP_TEST_NAME}.map

#MAP_TRAIN_NAME=pdb_3bcu_train
#MAP_TEST_NAME=pdb_3bcu_test
MAP_TRAIN_NAME=tiny_train
MAP_TEST_NAME=tiny_test
#MAP_TRAIN_NAME=l0_1pct_train
#MAP_TEST_NAME=l0_1pct_test
#MAP_TRAIN_NAME=l0_split_10_train
#MAP_TEST_NAME=l0_split_10_test

#MAP_TRAIN_NAME=l0_split_10
#MAP_TEST_NAME=l0_split_90
MAP_TRAIN_PATH=/lus/scratch/avose/data/map/${MAP_TRAIN_NAME}.map
MAP_TEST_PATH=/lus/scratch/avose/data/map/${MAP_TEST_NAME}.map

#Please find some new datsets out on Lustre on Osprey to use with the new ensemble approach.  I actually chose to make 5 different training files, each one is 5% of the total bindingdb, for a 25% total train and 75% total testing split.  You can find the files at the following paths:

#-rw-r--r--. 1 avose criemp 9.3M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_a.map
#-rw-r--r--. 1 avose criemp 8.6M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_b.map
#-rw-r--r--. 1 avose criemp 9.0M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_c.map
#-rw-r--r--. 1 avose criemp 8.9M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_d.map
#-rw-r--r--. 1 avose criemp 9.6M Aug  6 07:52 /lus/scratch/avose/data/map/bindingdb_2019m4_5of25pct_e.map
#-rw-r--r--. 1 avose criemp 149M Jun 23 07:06 /lus/scratch/avose/data/map/bindingdb_2019m4_75.map



NODES=4 #nodes total
NP=32 #processes total
NT=2 #threads per rank
BS=4 #batch size per rank
LR0=0.000001
MLP_LATENT=32,32
MLP_LAYERS=2,2
GNN_LAYERS=8,8
NUM_FEATURES=16,16
MODE=classification
#DATA_MODE=none,atoms+edges
EPOCHS=10

TEMP_DIR=${SCRATCH}/temp/ensemble-mldock-train_${MAP_TRAIN_NAME}-test_${MAP_TEST_NAME}-np_${NP}-lr_${LR0}-${GNN_LAYERS}_layer-${MLP_LATENT}x${MLP_LAYERS}-bs_${BS}-epochs_${EPOCHS}-nf_${NUM_FEATURES}_fresh
rm -rf $TEMP_DIR
mkdir -p ${TEMP_DIR}
cp ./* ${TEMP_DIR}/
cd ${TEMP_DIR}
export SLURM_WORKING_DIR=${TEMP_DIR}

echo
echo "Settings:"
pwd
ls
echo

echo "Running..."
date
time srun --mem=0 -p spider --label -N ${NODES} -n ${NP} --ntasks-per-socket=4 --ntasks-per-node=8 --accel-bind=g,v --cpu_bind=none -u --exclusive python mldock_gnn.py \
    --map_train ${MAP_TRAIN_PATH} \
    --map_test ${MAP_TEST_PATH} \
    --batch_size ${BS} \
    --mlp_latent ${MLP_LATENT} \
    --mlp_layers ${MLP_LAYERS} \
    --gnn_layers ${GNN_LAYERS} \
    --lr_init ${LR0} \
    --use_clr True \
    --hvd True \
    --num_features ${NUM_FEATURES} \
    --data_threads ${NT} \
    --mode "classification" \
    --epochs ${EPOCHS} 2>&1 |& tee myrun.out



#
#    --restore="/lus/scratch/jbalma/temp/mldock-train_l0_split_10-test_l0_1pct_test-np_64-lr_0.000001-8,8_layer-32,32x2,2-bs_4-epochs_10-nf_16,16_resumed_4/checkpoints/model.ckpt" \
#/lus/scratch/jbalma/temp/mldock-train_l0_split_10-test_l0_1pct_test-np_64-lr_0.000001-8,8_layer-32,32x2,2-bs_4-epochs_1000-nf_16,16_fresh/
#/lus/scratch/jbalma/temp/mldock-train_l0_split_10-test_l0_1pct_test-np_64-lr_0.000001-8,8_layer-32,32x2,2-bs_4-epochs_1000-nf_16,16_resumed_1


#--restore="/lus/sonexion/jbalma/temp/mldock-train_l0_1pct_train-test_l0_1pct_test-np_32-lr_0.000001-8,8_layer-32,32x2,2-bs_4-epochs_1000-nf_16,16_fresh/checkpoints/model.ckpt" \
#    --restore="/lus/scratch/jbalma/temp/mldock_10pct_train_90pct_test_runs/mldock-train_l0_split_10-test_l0_split_90-np_64-lr_0.000001-8,8_layer-32,32x2,2-bs_4-epochs_1000-nf_16,16_resumed_epoch18_86acc/checkpoints/model.ckpt"
#--restore="/lus/scratch/jbalma/temp/mldock-train_l0_split_10-test_l0_split_90-np_64-lr_0.000001-8,8_layer-32,32x2,2-bs_4-epochs_1000-nf_16,16_resumed_epoch14_85acc/checkpoints/model.ckpt" \
# --restore="/lus/scratch/jbalma/temp/mldock-train_l0_split_10-test_l0_split_10_test-np_64-lr_0.000001-8,8_layer-32,32x2,2-bs_4-epochs_1000-nf_16,16_resumed_epoch6_77acc/checkpoints/model.ckpt" 
#--restore="/lus/scratch/jbalma/temp/mldock-train_l0_split_10-test_l0_split_10_test-np_32-lr_0.000001-8,8_layer-32,32x2,2-bs_4-epochs_1000-nf_16,16_resumed_epoch3_70acc/checkpoints/model.ckpt" \
#--restore="/lus/scratch/jbalma/temp/mldock-train_l0_split_10-test_l0_split_10_test-np_32-lr_0.000001-8,8_layer-32,32x2,2-bs_4-epochs_1000-nf_16,16_fresh/checkpoints/model.ckpt" \
#    --restore="/lus/scratch/jbalma/temp/mldock-l0_split_10_train-l0_split_10_test-np_64-lr_0.000001-8,8_layer-32,32x2,2-bs_4-epochs_1000-datamode_protein,ligand-nf_16,16/checkpoints/model.ckpt" \
#    --epochs ${EPOCHS} 2>&1 |& tee myrun.out
date
#--mode regression \


echo "Done..."



