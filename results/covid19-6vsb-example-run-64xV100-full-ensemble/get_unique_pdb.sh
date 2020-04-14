#!/bin/bash
#FILENAME=/lus/scratch/avose/data/map/tiny_train.map
#FILENAME=/lus/scratch/avose/data/map/tiny_test.map
#FILENAME=/lus/scratch/avose/data/map/l0_1pct_train.map
#FILENAME=/lus/scratch/avose/data/map/l0_1pct_test.map
#FILENAME=/lus/scratch/avose/data/map/l0_split_10.map
#FILENAME=/lus/scratch/avose/data/map/l0_split_90.map
#FILENAME=/lus/scratch/avose/data/map/bindingdb_2019m4_75.map
FILENAME=/lus/scratch/avose/data/map/bindingdb_2019m4
echo "number items:"
cat ${FILENAME} | wc -l

echo "Unique PDB IDs in ${FILENAME}"
awk '{ a[$2]++ } END { for (b in a) { print b } }' ${FILENAME} | wc -l

echo "Unique Ligand IDs in ${FILENAME}"
awk '{ a[$3]++ } END { for (b in a) { print b } }' ${FILENAME} | wc -l
