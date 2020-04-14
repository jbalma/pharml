#!/bin/bash
#Script to tell you the number of ligand-protein 
#pairs in a map file, and counts of unique PDB IDs and Ligands

FILENAME=/lus/scratch/avose/data/map/bindingdb_2019m4_75.map

echo "number items in ${FILENAME}:"
cat ${FILENAME} | wc -l

echo "Unique PDB IDs in ${FILENAME}"
awk '{ a[$2]++ } END { for (b in a) { print b } }' ${FILENAME} | wc -l

echo "Unique Ligand IDs in ${FILENAME}"
awk '{ a[$3]++ } END { for (b in a) { print b } }' ${FILENAME} | wc -l
