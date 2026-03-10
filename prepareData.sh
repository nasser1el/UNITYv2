#!/usr/bin/env bash
# prepareData.sh - download GWAS files and run preprocessing
# run from the repo root: bash prepareData.sh
# PGC files (SCZ/BIP) need to be downloaded manually first (see README)

mkdir -p data
cd data

# Height - Wood et al. 2014 (GIANT)
wget -q --show-progress \
    "https://portals.broadinstitute.org/collaboration/giant/images/0/01/GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz"
gunzip GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt.gz

# BMI - Locke et al. 2015 (GIANT)
wget -q --show-progress \
    "https://portals.broadinstitute.org/collaboration/giant/images/1/15/SNP_gwas_mc_merge_nogc.tbl.uniq.gz"
gunzip SNP_gwas_mc_merge_nogc.tbl.uniq.gz

# HDL / LDL - Willer et al. 2013 (GLGC)
wget -q --show-progress \
    "http://csg.sph.umich.edu/willer/public/lipids2013/jointGwasMc_HDL.txt.gz"
wget -q --show-progress \
    "http://csg.sph.umich.edu/willer/public/lipids2013/jointGwasMc_LDL.txt.gz"
gunzip jointGwasMc_HDL.txt.gz jointGwasMc_LDL.txt.gz

# SCZ / BIP: download manually from https://pgc.unc.edu/for-researchers/download-results/
# and place in data/ before running prepGWAS.R:
#   PGC3_SCZ_wave3.european.autosome.public.v3.vcf.tsv
#   pgc-bip2021-all.vcf.tsv

cd ..
Rscript prepGWAS.R
