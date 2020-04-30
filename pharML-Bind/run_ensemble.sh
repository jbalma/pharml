#!/bin/bash
#Add the PDB IDs which you have generated results for here
pdbid_list="6vyb 6vsb 6lzg 6m3m 6w02 6y2e 6y84"

for PDB_ID in $pdbid_list;
do
    INFERENCE_RESULTS="../../results-covid19/ncnpr-all/pharml-bind-covid-${PDB_ID}-np-32-lr0.000000001-bs8-ncnpr-all/results/covid19/inference"
    OUTPUT_DIR=ensemble5x_covid19_results_${PDB_ID}_ncnpr_full
    mkdir $OUTPUT_DIR
    cd $OUTPUT_DIR
    cp ../ensemble-rank.py .
    cp ../chemio.py .
    MAPS="${INFERENCE_RESULTS}/model_0/combined_predictions_${PDB_ID}_inference_model0.map:${INFERENCE_RESULTS}/model_1/combined_predictions_${PDB_ID}_inference_model1.map:${INFERENCE_RESULTS}/model_2/combined_predictions_${PDB_ID}_inference_model2.map:${INFERENCE_RESULTS}/model_3/combined_predictions_${PDB_ID}_inference_model3.map:${INFERENCE_RESULTS}/model_4/combined_predictions_${PDB_ID}_inference_model4.map"

    python ensemble-rank.py --out ensemble5x_25pcttrain.txt --maps ${MAPS} 

    cat ${INFERENCE_RESULTS}/model_*/combined_predictions_${PDB_ID}_inference_model*.map > ./ensemble5x_all.out

    cat ./ensemble5x_all.out | sort | uniq --all-repeated --unique | uniq -c | sort > ensemble5x_final_ranked_${PDB_ID}.out

    cat ./ensemble5x_final_ranked_${PDB_ID}.out | grep "4 .*.pred:1" | sort > ensemble5x_final_top_${PDB_ID}.out

    cat ensemble5x_final_top_${PDB_ID}.out | sed 's/.*.lig\/lig//' | sed 's/.lig.*.//' > ensemble5x_top_${PDB_ID}.txt

    echo "<html><body>PharML Top Ranked Compounds" > ensemble5x_top_${PDB_ID}.html
    EMBED_LINE="<object data=\"https:\/\/www.ebi.ac.uk\/chembl\/embed/#compound_report_card/${CHEMBL}/name_and_classification\" width=\"100%\" height=\"100%\"></object>"

    IFS=$'\n'       # make newlines the only separator
    set -f          # disable globbing

    for CHEMBL in $(cat < "ensemble5x_top_${PDB_ID}.txt"); do
        EMBED_LINE="<object data=\"https://www.ebi.ac.uk/chembl/embed/#compound_report_card/${CHEMBL}/name_and_classification\" width=\"100%\" height=\"100%\"></object>"
      echo "tester: ${EMBED_LINE}"
      echo ${EMBED_LINE} >> ensemble5x_top_${PDB_ID}.html
    done
    echo "<\\body><\\html>" >> ensemble5x_top_${PDB_ID}.html
    cd ../
done
#Format needs to look like this in final HTML file
#<object data="https://www.ebi.ac.uk/chembl/embed/#compound_report_card/CHEMBL603/name_and_classification" width="100%" height="100%"></object>


