#!/bin/bash
INFERENCE_RESULTS="../results/covid19-6vsb-example-run-64xV100-full-ensemble/results/covid19_2020-Apr04-1585894036/inference"

MAPS="${INFERENCE_RESULTS}/model_0/combined_predictions_6vsb-bindingdb_inference_model0.map:${INFERENCE_RESULTS}/model_1/combined_predictions_6vsb-bindingdb_inference_model1.map:${INFERENCE_RESULTS}/model_2/combined_predictions_6vsb-bindingdb_inference_model2.map:${INFERENCE_RESULTS}/model_3/combined_predictions_6vsb-bindingdb_inference_model3.map:${INFERENCE_RESULTS}/model_4/combined_predictions_6vsb-bindingdb_inference_model4.map"

python ensemble.py --out ensemble5x_25pcttrain.txt --maps ${MAPS} 
#combined_predictions_6vsb-bindingdb_inference_model0.map
cat ${INFERENCE_RESULTS}/model_*/combined_predictions_6vsb-bindingdb_inference_model*.map > ./ensemble5x_all.out

cat ./ensemble5x_all.out | sort | uniq --all-repeated --unique | uniq -c | sort > ensemble5x_final_ranked_6vsb.out

cat ./ensemble5x_final_ranked_6vsb.out | grep "4 .*.pred:1" | sort > ensemble5x_final_top_6vsb.out

cat ensemble5x_final_top_6vsb.out | sed 's/.*.lig\/lig//' | sed 's/.lig.*.//' > ensemble5x_top_6vsb_chembl.txt

echo "<html>\n<body>\n" > ensemble5x_top_6vsb_chembl.html
EMBED_LINE="<object data=\"https:\/\/www.ebi.ac.uk\/chembl\/embed/#compound_report_card/${CHEMBL}/name_and_classification\" width=\"100%\" height=\"100%\"></object>"

IFS=$'\n'       # make newlines the only separator
set -f          # disable globbing
for CHEMBL in $(cat < "ensemble5x_top_6vsb_chembl.txt"); do
  EMBED_LINE="<object data=\"https://www.ebi.ac.uk/chembl/embed/#compound_report_card/${CHEMBL}/name_and_classification\" width=\"100%\" height=\"100%\"></object>"
  echo "tester: ${EMBED_LINE}"
  echo ${EMBED_LINE} >> ensemble5x_top_6vsb_chembl.html
done
echo "<\\body>\n<\\html>\n" >> ensemble5x_top_6vsb_chembl.html

#Format needs to look like this in final HTML file
#<object data="https://www.ebi.ac.uk/chembl/embed/#compound_report_card/CHEMBL603/name_and_classification" width="100%" height="100%"></object>


