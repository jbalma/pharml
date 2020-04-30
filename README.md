# PharML
PharML is a framework for predicting compound affinity for protein structures. It utilizes a novel Molecular-Highway Graph Neural Network (MH-GNN) architecture based on state-of-the-art techniques in deep learning. This repository contains the visualization, preprocessing, training, and inference code written in Python and C. In addition, we provide an ensemble of pre-trained models which can readily be used for quickly generating rank-ordered predictions of compound affinity relative to a given target. DISCLAIMER: Compounds predicted by PharML.Bind should not be used without consulting a doctor or pharmacist - all results should be considered unverified and used only as a starting point for further investigation. Use at your own risk!

Setup
==============================

1) Edit the conda environment script to reflect your system configuration

    vim tools/environments/setup_conda_env.sh

    -> Ensure you have cudatoolkit and appropriate drivers to match the conda environment
    -> See README under /tools/environments for more details

2) Activate the conda environment

    source activate /path/to/conda-pharml-env

    -> Install the following packages or do pip install -r tools/requirements.txt

    [todo: list package version requirements and put those in tools/environments/requirements.txt]

3) Preprocess the Dataset

    -> The examples directory is setup with scripts for
        a) preprocessing of COVID-19 structure PDB, and BindingDB FDA-approved compounds in SDF format
        b) preprocessing of the full BindingDB dataset
        c) visualization of structures with associated compounds


    -> After preprocessing completes, you will have a directory containing
        -> data/lig: the ligand graph files
        -> data/nhg: the protein neighborhood graph (NHG) files
        -> data/pdb: the raw pdb files used to generate ligands and NHG
        -> data/map: the map file used for inference that specifies the ligand-to-target tests which will be tested


Running
==============================

4) Test Inference Across example map file

    -> Launch with the following command to test against the COVID-19 6VSU structure and bindingDB's FDA-approved compound list generated in step 3
        
        python mldock_gnn.py \
            --map_test ../datasets/covid19/map/6VSU.map \
            --batch_size_test 16 \
            --mlp_latent ${MLP_LATENT} \
            --mlp_layers ${MLP_LAYERS} \
            --gnn_layers ${GNN_LAYERS} \
            --hvd True \
            --num_features ${NUM_FEATURES} \
            --data_threads 2 \
            --mode classification \
            --inference_only True
            --restore=../pretrain-models/mh-gnnx5-ensemble/ensemble_member_${n}/checkpoints/model0.ckpt" \
            --inference_out covid19_inference_${MAP_TEST_NAME}.out \
            --epochs 1 2>&1 |& tee log-covid19-${MAP_TEST_NAME}.out

    -> Using the --inference_out options tells PharML.Bind to save the outputs to disk, indexed by the compound ID




5) Run Inference with each ensemble member to generate rank-ordered compound set


    -> You can also use the runit_inference.sh script as follows:

        Note: This will launch one pre-trained ensemble member each iteration indexed by n. When looping over each memeber, it generates predictions for the compound set on target PDB ID set by MAP_TEST_NAME. MAP_TEST_NAME is the index of the structure list which by default is set by PDB_ID_LIST="6LZG 6VSB 6LU7"

        salloc -N 8 -n 64 --ntasks-per-node=8 -t 24:00:00

        #Start CUDA MPS Server on each of the Dense GPU nodes
        time srun --cpu_bind=none -p spider -C V100 -l -N 8 --ntasks-per-node=1 -u ./restart_mps.sh 2>&1 |& tee mps_result.txt

        #Start the inference run on a single model
        srun -c 4 --hint=multithread -C V100 -p spider -l -N 8 -n 64 --ntasks-per-node=8 --ntasks-per-socket=4 -u --cpu_bind=none python mldock_gnn.py \
        --map_test ../datasets/covid19/map/${MAP_TEST_NAME}.map \
        --batch_size_test 16 \
        --mlp_latent ${MLP_LATENT} \
        --mlp_layers ${MLP_LAYERS} \
        --gnn_layers ${GNN_LAYERS} \
        --hvd True \
        --num_features ${NUM_FEATURES} \
        --data_threads 2 \
        --mode classification \
        --inference_only \
        --restore=../pretrain-models/mh-gnnx5-ensemble/ensemble_member_${n}/checkpoints/model0.ckpt" \
        --inference_out covid19_inference_${MAP_TEST_NAME}.out \
        --epochs 1 2>&1 |& tee covid19-${MAP_TEST_NAME}.out


    -> You should see the following output if you use SLURM workload manager to launch via srun command using the runit_inference.sh script:

        "Starting COVID-19 Structure Inference for 6VSB, 6LZG, 6LU7 for ensemble member 0..."
        "========================================================================================"
        "--> 6LZG: Spike receptor-binding domain complexed with its receptor ACE2: https://www.rcsb.org/structure/6LZG"
        "--> 6VSB: Prefusion 2019-nCoV spike glycoprotein with a single receptor-binding domain up: https://www.rcsb.org/structure/6vsb"
        "--> Starting at: 4/3/2020 12:00:00 PM CST"

 
    -> Next, you will see PharML.Bind initialize and tensorflow will print some logging info

        0: WARNING:tensorflow:From mldock_gnn.py:57: The name tf.set_random_seed is deprecated. Please use tf.compat.v1.set_random_seed instead.
        0: Initialization of horovod complete...
        0: Rank 0  is saving inference output to model0_dataset_inference_0.map
        0: ----------------------------------------
        0: PharML.Bind-GNN: Version 1.0.1 - Framework for Open Therapeutics with Graph Neural Networks.
        0: ----------------------------------------
        0: ============================================================================================
        0: ----------------------------------------
        0:   Developed by
        0: ----------------------------------------
        0: ----------------------------------------
        0:       Jacob Balma: jb.mt19937@gmail.com
        0: ----------------------------------------
        0: ----------------------------------------
        0:       Aaron Vose:  avose@aaronvose.net
        0: ----------------------------------------
        0: ----------------------------------------
        0:       Yuri Petersen: yuripeterson@gmail.com
        0: ----------------------------------------
        0: 
        0: ----------------------------------------
        0: This work is supported by collaboration with Cray, Inc, Medical University of South Carolina (MUSC) and Hewlett Packard Enterprise (HPE). 
        0: ----------------------------------------
        0: ----------------------------------------
        0: ============================================================================================
        0: ----------------------------------------
        0: Namespace(batch_size=2, batch_size_test=2, data_threads=4, debug=True, epochs=1, gnn_layers='5,5', hvd=True, inference_only=True, inference_out='model0_dataset_inference.map', lr_init=0.01, map_test='/lus/scratch/jbalma/DataSets/Binding/mldock/tools/covid19-fda-bindingdb/data-6vsb-bindingdb-fda/map/dataset.map', map_train='/lus/scratch/jbalma/avose_backup/data/map/l0_1pct_train.map', mlp_latent='32,32', mlp_layers='2,2', mode='classification', num_features='16,16', plot_history=False, restore='./pretrained-models/mh-gnnx5-ensemble/model_0/checkpoints/model0.ckpt', use_clr=False, use_fp16=False)
        0: ----------------------------------------
        0: Loading data.
        0: ----------------------------------------
        0: Loading map file as distributed dataset for 8 ranks:
        0:   Map file:          /lus/scratch/jbalma/data/6vsb-bindingdb-fda/map/l0_1pct_train.map
        0:   Map items:         23608
        0:   Worker chunk size: 2951
        0:   Number of items:   2951
        0:   Sum(target[0]):    1422.000000
        0:   Avg(target[0]):    0.481871
        0:   Total load time:   0.4
        0: Done processing map file.
        0: Loading map file as distributed dataset for 8 ranks:
        0:   Map file:          /lus/scratch/jbalma/data/6vsb-bindingdb-fda/map/dataset.map
        0:   Map items:         440
        0:   Worker chunk size: 55
        0:   Number of items:   55
        0:   Sum(target[0]):    55.000000
        0:   Avg(target[0]):    1.000000
        0:   Total load time:   0.0
        0: Done processing map file.

    -> When inference finishes over all ensemble members, you will find the results stored to disk (if --inference_out was set) in the pharml/results/${MAP_TEST_NAME} directory


Evaluate and Visualize
==============================

6) Evaluate the results

    -> Using the scripts provided in pharml/results, we can evaluate the rank-ordered compound set generated in the previous step

    "[TODO: provide example of using the ensemble summary tool]"

7) Visualize the target with predicted compounds

    -> Using mlvoxelizer, we can visualize the compounds, proteins, or both relative to one another (when active-site training was used)

    "[TODO: Fill in an example of using mlvoxelizer for visualization of a covid-19 structure]"


