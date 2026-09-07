READS=/shares/swabseq/kaiser2-tecan/Manual_TECAN_REVELO_D03_CFA_RES_BPK_00683_0139121_Swab_D03_UDI_D_S20_R1_001.fastq.gz
PAIR=/shares/swabseq/kaiser2-tecan/Manual_TECAN_REVELO_D03_CFA_RES_BPK_00683_0139121_Swab_D03_UDI_D_S20_R2_001.fastq.gz
LIBRARY=/shares/swabseq/db/databases/metagenomic_data/NT_Viral_Data/AllNucleotide_090125_2697049_filtered+2903_rep_COVID_fixed_COVID_naming.fa
DATABASE=/shares/swabseq/db/databases/metagenomic_data/01Dec2025-AIMetadata-Step3
OUTPUT_DIR=notebooks/d03_results_kaiser2

trident \
    --reads $READS --pair $PAIR \
    --library $LIBRARY --database $DATABASE \
    --output $OUTPUT_DIR \
    --save-figs

trident accession \
    --reads $READS --pair $PAIR \
    --library $LIBRARY --database $DATABASE \
    --output $OUTPUT_DIR \
    --n-steps 2000 \
    --save-figs

trident check \
    --reads $READS --pair $PAIR \
    --library $LIBRARY \
    --output $OUTPUT_DIR \
    --save-figs