#!/bin/bash
#ensemble5x_covid19_results_6lzg_chembl_phase1234/
DATA_NAME=chembl_phase1234
ENS_DIR_NAME=ensemble5x_covid19_results
echo "Number of total compounds across all PDB IDs:"
cat ${ENS_DIR_NAME}_*_${DATA_NAME}/top-compounds-*-ID-only-ranked-fp.out | wc -l
#First generate a file which contains only the overlapping compounds which occur across all PDBIDs
echo "Number of total overlapping compounds which occur across all PDB IDs:"
cat ${ENS_DIR_NAME}_*_${DATA_NAME}/top-compounds-*-ID-only-ranked-fp.out | sort | uniq -c | sort | grep " 7 " | awk '{printf "%s\n", $2}' > overlap_only.txt
cat overlap_only.txt | wc -l

pdbid_list="6vyb 6vsb 6lzg 6m3m 6w02 6y2e 6y84"

for PDB_ID in $pdbid_list;
do

    NO_OL_OUT=${PDB_ID}_no_overlap.txt
    HTML_OUTPUT=ensemble5x_top_${PDB_ID}_uniq.html
    HTML_OUTPUT2=top_compounds_${PDB_ID}_rawranking.html

    IFS=$'\n'       # make newlines the only separator
    set -f          # disable globbing

    #Make a per-PDB file that contains only the non-overlapping CHEMBL IDs, but retain the ranking order
    rm ${NO_OL_OUT}
    date > ${NO_OL_OUT}
    cat ${NO_OL_OUT}
    sleep 2
    num_ol=0
    num_uniq=0
    total_drugs=$(( `cat ${ENS_DIR_NAME}_${PDB_ID}_${DATA_NAME}/top-compounds-${PDB_ID}-ID-only-ranked-fp.out | wc -l` ))
    total_overlap=$(( `cat overlap_only.txt | wc -l` ))

    cat overlap_only.txt ${ENS_DIR_NAME}_${PDB_ID}_${DATA_NAME}/top-compounds-${PDB_ID}-ID-only-ranked-fp.out | sort > ${PDB_ID}_sorted.txt
    cat ${PDB_ID}_sorted.txt | uniq -u > ${PDB_ID}_uniq_only.txt
    total_uniq=$(( `cat ${PDB_ID}_uniq_only.txt | wc -l` ))
    
    head ${ENS_DIR_NAME}_${PDB_ID}_${DATA_NAME}/top-compounds-${PDB_ID}-ID-only-ranked-fp.out > ${PDB_ID}_head_top_ranked.txt
    for CHEMBL in $(cat < ${ENS_DIR_NAME}_${PDB_ID}_${DATA_NAME}/top-compounds-${PDB_ID}-ID-only-ranked-fp.out); do
        #echo "CHEMBL=" $CHEMBL
        for OC in $(cat < ${PDB_ID}_uniq_only.txt); do
            if [ "$OC" == $CHEMBL ]; then
                EMBED_LINE="${CHEMBL}"
                #echo "  -> tester: ${EMBED_LINE}"
                echo ${EMBED_LINE} >> ${NO_OL_OUT}
                let "num_uniq=num_uniq+1"
                #break
            else
                let "num_ol=num_ol+1"
            fi
        done
        #else
            #    let "num_ol=num_ol+1"
        

        #ol_count=0
        #for OC in $(cat < overlap_only.txt); do
        #    if [ "$OC" == $CHEMBL ]; then
                #echo "Overlapping ID"
                #echo "  -> Match! OC ${OC}"
        #        unset EMBED_LINE
        #        let "num_ol=num_ol+1"
        #        let "ol_count=ol_count+1"
                #break
        #    else
                #echo "  -> Fail to find match! OC ${OC}"
                #echo "Strings NOT equal"
        #        EMBED_LINE="${CHEMBL}"
                #echo "  -> tester: ${EMBED_LINE}"
                #echo ${EMBED_LINE} >> ${NO_OL_OUT}
        #        let "num_uniq=num_uniq+1"
                #unset EMBED_LINE
                #break
        #    fi
        #done

        #echo "overlap counter = $num_ol / $total_overlap"
        #if [ $num_ol == $num_uniq ]; then
            #echo "All overlap, ol_count=$ol_count"
        #    echo ${EMBED_LINE} >> ${NO_OL_OUT}
        #else
            #echo ${EMBED_LINE} >> ${NO_OL_OUT}
        #    echo "Done checking overlap for $CHEMBL, ol_count=$ol_count"
        #fi
        #unset EMBED_LINE
        
    done
    echo "PDB ID: ${PDIB_ID} had uniq: ${num_uniq} compounds; out of total: ${total_drugs}; removed ${num_ol} overlapping compounds."
    echo " -> expecting total uniq CHEMBL for ${PDB_ID} to be $total_uniq"
    #cat ${NO_OL_OUT} | uniq -u > ${NO_OL_OUT}

    #Generate an HTML file with the non-overlapping compounds per-PDB
    echo "<html><body>PharML Top Ranked Compounds for ${PDB_ID} - Overlap Removed" > ${HTML_OUTPUT}
    EMBED_LINE="<object data=\"https:\/\/www.ebi.ac.uk\/chembl\/embed/#compound_report_card/${CHEMBL}/name_and_classification\" width=\"100%\" height=\"100%\"></object>"

    IFS=$'\n'       # make newlines the only separator
    set -f          # disable globbing
    i=1
    for CHEMBL in $(cat < ${NO_OL_OUT}); do
        EMBED_LINE="<object data=\"https://www.ebi.ac.uk/chembl/embed/#compound_report_card/${CHEMBL}/name_and_classification\" width=\"100%\" height=\"100%\"></object>"
        #echo "tester: ${EMBED_LINE}"
        echo "Compound ranked #${i}: " >> ${HTML_OUTPUT}
        echo ${EMBED_LINE} >> ${HTML_OUTPUT}
        
        let "i=i+1"
    done
    echo "</body></html>" >> ${HTML_OUTPUT}


#Generate an HTML file with the non-overlapping compounds per-PDB
    echo "<html><body>PharML Top Ranked Compounds for ${PDB_ID} - Raw Ranking" > ${HTML_OUTPUT2}
    EMBED_LINE="<object data=\"https:\/\/www.ebi.ac.uk\/chembl\/embed/#compound_report_card/${CHEMBL}/name_and_classification\" width=\"100%\" height=\"100%\"></object>"

    IFS=$'\n'       # make newlines the only separator
    set -f          # disable globbing
    i=1
    for CHEMBL in $(cat < ${PDB_ID}_head_top_ranked.txt); do
        EMBED_LINE="<object data=\"https://www.ebi.ac.uk/chembl/embed/#compound_report_card/${CHEMBL}/name_and_classification\" width=\"100%\" height=\"100%\"></object>"
        #echo "tester: ${EMBED_LINE}"
        echo "Compound ranked #${i}: " >> ${HTML_OUTPUT2}
        echo ${EMBED_LINE} >> ${HTML_OUTPUT2}
        let "i=i+1"
    done
    echo "</body></html>" >> ${HTML_OUTPUT2}


done




