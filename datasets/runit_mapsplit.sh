#!/bin/bash


#Point to a map file (dataset.map) and split it into train and test set

#    parser.add_argument('--map', type=str,   required=True,   help='Path to map file.')
#    parser.add_argument('--out', type=str,   default="split", help='Output file prefix.')
#    parser.add_argument('--pct', type=float, default=20.0,    help='Split point (percent).')

source ../mldock-gnn/config_cuda10.sh
unset PYTHONPATH
module rm PrgEnv-cray

INSTALL_DIR=/lus/scratch/jbalma/condenv-cuda10-pharml

#conda create -y --prefix $INSTALL_DIR python=3.6 cudatoolkit=10.0 cudnn
source activate $INSTALL_DIR/
export PATH=${INSTALL_DIR}/bin:${PATH} #/home/users/${USER}/.local/bin:${PATH}
export PYTHONPATH="./mlvoxelizer:$PYTHONPATH"

#MAP_FILE=/lus/scratch/jbalma/avose_backup/data/map/bindingdb_2019m4_75.map
#MAP_FILE=/lus/scratch/jbalma/avose_backup/data/map/bindingdb_2019m4_5of31of75.map
MAP_FILE=/lus/scratch/jbalma/DataSets/Binding/bindingdb_2019m4/data/map/bindingdb_2019m4_69of75.map
#BindingDB used for training initial 5 ensemble members used 25% of data, 842766 PLP (bindingdb_2019m4_25.map)
#BindingDB used for validating initial 5 ensemble members used remaining 75% of data 2708400 PLP (bindingdb_2019m4_75.map)

#We want to now use another 25% of BindingDB, but we want to draw it from that 75% used in validation
#So, we want the resulting map to take in the validation map, and output a new map of 842766 PLP to train on
#So 2708400*x=842766; x=31.12%
#That will give us our new training set that should contain 842766 PLP
python map_split.py --map ${MAP_FILE} --pct 1.0 --out bindingdb_2019m4_1of69of75
#Now we want to take 5% of the that file
#$ cat bindingdb_2019m4_31of75_a.map | wc -l
#843980
#$ cat bindingdb_2019m4_31of75_b.map | wc -l
#1850944
#now we want x*843980=167670
#512776
#now we want x*512776=167670, x=32.698
#351884
#now we want x*351884=167670, x=47.649
#python map_split.py --map bindingdb_2019m4_5of31of75_b.map --pct 47.649 --out bindingdb_2019m4_5of31of75
#now split again using new length of b
#x*681824=
#python map_split.py --map bindingdb_2019m4_31of75_b.map --pct 9.0586 --out bindingdb_2019m4_9of31of75
