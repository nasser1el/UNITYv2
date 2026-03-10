#!/usr/bin/env bash
# runUNITY.sh - run all analyses (real GWAS + simulations)
# run from the repo root after prepareData.sh: bash runUNITY.sh
# h2 and rho values are from published LDSC estimates (see README)

# Height / BMI
# NOTE: N values are median per-SNP sample sizes, not total cohort sizes
M=$(wc -l < data/height_betahat.txt)
python UNITYv2.py \
    --file1 data/height_betahat.txt --file2 data/bmi_betahat.txt \
    --N1 251631 --N2 339224 --H1 0.239 --H2 0.157 --rho 0.085 \
    --M $M --ITS 3000 --method mh --seed 486 --id height_bmi_mh

python UNITYv2.py \
    --file1 data/height_betahat.txt --file2 data/bmi_betahat.txt \
    --N1 251631 --N2 339224 --H1 0.239 --H2 0.157 --rho 0.085 \
    --M $M --ITS 3000 --method gibbs --seed 486 --id height_bmi_gibbs


# SCZ / BIP
# NOTE: N_eff = 4 / (1/N_cases + 1/N_controls); rho = 0.68 from Anttila 2018
# needs more iterations to converge from the prior given small N_eff
M=$(wc -l < data/scz_betahat.txt)
python UNITYv2.py \
    --file1 data/scz_betahat.txt --file2 data/bip_betahat.txt \
    --N1 58749 --N2 50981 --H1 0.24 --H2 0.20 --rho 0.68 \
    --M $M --ITS 5000 --method mh --seed 486 --id scz_bip_mh

python UNITYv2.py \
    --file1 data/scz_betahat.txt --file2 data/bip_betahat.txt \
    --N1 58749 --N2 50981 --H1 0.24 --H2 0.20 --rho 0.68 \
    --M $M --ITS 5000 --method gibbs --seed 486 --id scz_bip_gibbs


# HDL / LDL
M=$(wc -l < data/hdl_betahat.txt)
python UNITYv2.py \
    --file1 data/hdl_betahat.txt --file2 data/ldl_betahat.txt \
    --N1 93561 --N2 89138 --H1 0.12 --H2 0.12 --rho -0.10 \
    --M $M --ITS 3000 --method mh --seed 486 --id hdl_ldl_mh

python UNITYv2.py \
    --file1 data/hdl_betahat.txt --file2 data/ldl_betahat.txt \
    --N1 93561 --N2 89138 --H1 0.12 --H2 0.12 --rho -0.10 \
    --M $M --ITS 3000 --method gibbs --seed 486 --id hdl_ldl_gibbs


# Simulations
# validation: Height/BMI parameter regime, MH then Gibbs
# true p = (0.95, 0.02, 0.02, 0.01)
python UNITYv2.py --sim \
    --A00 0.95 --A10 0.02 --A01 0.02 --A11 0.01 \
    --N1 251631 --N2 339224 --H1 0.239 --H2 0.157 --rho 0.085 \
    --M 100000 --ITS 5000 --method mh --seed 486 --id sim_validation_mh

python UNITYv2.py --sim \
    --A00 0.95 --A10 0.02 --A01 0.02 --A11 0.01 \
    --N1 251631 --N2 339224 --H1 0.239 --H2 0.157 --rho 0.085 \
    --M 100000 --ITS 5000 --method gibbs --seed 486 --id sim_validation_gibbs

# overlap demo: rho0=0.3, not modeled vs inferred
python UNITYv2.py --sim \
    --A00 0.95 --A10 0.02 --A01 0.02 --A11 0.01 \
    --N1 100000 --N2 100000 --H1 0.25 --H2 0.25 --rho 0.0 \
    --rho0_sim 0.3 \
    --M 100000 --ITS 5000 --method mh --seed 486 --id sim_overlap_ignored

python UNITYv2.py --sim \
    --A00 0.95 --A10 0.02 --A01 0.02 --A11 0.01 \
    --N1 100000 --N2 100000 --H1 0.25 --H2 0.25 --rho 0.0 \
    --rho0_sim 0.3 --infer_rho0 \
    --M 100000 --ITS 5000 --method mh --seed 486 --id sim_overlap_inferred
