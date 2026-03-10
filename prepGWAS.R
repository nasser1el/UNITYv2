#!/usr/bin/env Rscript
# prepGWAS.R - Preprocess GWAS summary statistics for UNITY v2
#
# Handles all three trait pairs:
#   1. Height (Wood 2014) / BMI (Locke 2015)
#   2. SCZ (PGC3) / BIP (PGC-BIP 2021)
#   3. HDL / LDL (GLGC Willer 2013)
#
# Usage: Rscript prepGWAS.R
#
# Expects raw GWAS files in the directory specified by dataDir (default: data/).
# Writes single-column betahat files to that same directory.

mafThreshold <- 0.05
pruneStep    <- 10       # approximate LD-pruning: retain every Nth SNP
dataDir      <- "data"


# Process one trait pair through the full preprocessing pipeline:
# overlap by rsID, allele alignment (with flip correction), MAF filter,
# removal of missing/zero-se rows, stride-based LD-pruning, and export.
# df1/df2 must have columns: rsid, a1, a2, beta, se, freq (optional), n (optional).
# trait1Name/trait2Name are used to name the output files.
processTraitPair <- function(df1, df2, trait1Name, trait2Name, pruneStep, mafThreshold) {

  cat("\n", strrep("-", 60), "\n")
  cat(" Processing:", trait1Name, "/", trait2Name, "\n")
  cat(strrep("-", 60), "\n")
  cat("  Trait 1 SNPs:", nrow(df1), "\n")
  cat("  Trait 2 SNPs:", nrow(df2), "\n")

  # Overlap by rsID
  shared <- intersect(df1$rsid, df2$rsid)
  cat("  Shared SNPs:", length(shared), "\n")

  df1 <- df1[df1$rsid %in% shared, ]
  df2 <- df2[df2$rsid %in% shared, ]

  df1 <- df1[!duplicated(df1$rsid), ]
  df2 <- df2[!duplicated(df2$rsid), ]

  df1 <- df1[order(df1$rsid), ]
  df2 <- df2[order(df2$rsid), ]
  stopifnot(all(df1$rsid == df2$rsid))
  cat("  After overlap + dedup:", nrow(df1), "\n")

  # Allele alignment: flip beta where a1/a2 are swapped; drop ambiguous
  same      <- (df1$a1 == df2$a1) & (df1$a2 == df2$a2)
  flipped   <- (df1$a1 == df2$a2) & (df1$a2 == df2$a1)
  ambiguous <- !(same | flipped)
  cat("  Same allele coding:", sum(same), "\n")
  cat("  Flipped (correcting):", sum(flipped), "\n")
  cat("  Ambiguous (dropping):", sum(ambiguous), "\n")

  df2$beta[flipped] <- -df2$beta[flipped]
  keep <- which(!ambiguous)
  df1 <- df1[keep, ]
  df2 <- df2[keep, ]

  # MAF filter (skipped if freq column is absent)
  if (!is.null(df1$freq) && !is.null(df2$freq)) {
    maf1 <- pmin(df1$freq, 1 - df1$freq)
    maf2 <- pmin(df2$freq, 1 - df2$freq)
    mafKeep <- which(maf1 >= mafThreshold & maf2 >= mafThreshold &
                     !is.na(maf1) & !is.na(maf2))
    df1 <- df1[mafKeep, ]
    df2 <- df2[mafKeep, ]
    cat("  After MAF filter:", nrow(df1), "\n")
  } else {
    cat("  No freq column - skipping MAF filter\n")
  }

  # Drop rows with missing or zero-se values
  validIdx <- which(!is.na(df1$beta) & !is.na(df2$beta) &
                    !is.na(df1$se)   & !is.na(df2$se)   &
                    df1$se > 0       & df2$se > 0)
  df1 <- df1[validIdx, ]
  df2 <- df2[validIdx, ]
  cat("  After removing invalid rows:", nrow(df1), "\n")

  # Approximate LD-pruning via stride
  idx <- seq(1, nrow(df1), by = pruneStep)
  df1 <- df1[idx, ]
  df2 <- df2[idx, ]
  cat("  After LD-prune (step =", pruneStep, "):", nrow(df1), "\n")

  # Export single-column betahat files
  outFile1 <- file.path(dataDir, paste0(trait1Name, "_betahat.txt"))
  outFile2 <- file.path(dataDir, paste0(trait2Name, "_betahat.txt"))
  write.table(df1$beta, file = outFile1, col.names = FALSE, row.names = FALSE, quote = FALSE)
  write.table(df2$beta, file = outFile2, col.names = FALSE, row.names = FALSE, quote = FALSE)

  cat("\n  Summary:\n")
  cat("  Final M:", nrow(df1), "\n")
  cat(" ", trait1Name, "- beta sd:", round(sd(df1$beta), 6),
      " | median N:", median(df1$n, na.rm = TRUE), "\n")
  cat(" ", trait2Name, "- beta sd:", round(sd(df2$beta), 6),
      " | median N:", median(df2$n, na.rm = TRUE), "\n")
  cat("  Output:", outFile1, ",", outFile2, "\n")

  invisible(list(m = nrow(df1),
                 n1 = median(df1$n, na.rm = TRUE),
                 n2 = median(df2$n, na.rm = TRUE)))
}


# Pair 1: Height (Wood 2014) / BMI (Locke 2015)
# Height columns: MarkerName Allele1 Allele2 Freq.Allele1.HapMapCEU b SE p N
# BMI columns:    SNP A1 A2 Freq1.Hapmap b se p N
heightFile <- file.path(dataDir, "GIANT_HEIGHT_Wood_et_al_2014_publicrelease_HapMapCeuFreq.txt")
bmiFile    <- file.path(dataDir, "SNP_gwas_mc_merge_nogc.tbl.uniq")

if (file.exists(heightFile) && file.exists(bmiFile)) {
  heightDf <- read.table(heightFile, header = TRUE, stringsAsFactors = FALSE)
  bmiDf    <- read.table(bmiFile,    header = TRUE, stringsAsFactors = FALSE)

  heightStd <- data.frame(
    rsid = heightDf$MarkerName,
    a1   = toupper(heightDf$Allele1),
    a2   = toupper(heightDf$Allele2),
    beta = heightDf$b,
    se   = heightDf$SE,
    freq = heightDf$Freq.Allele1.HapMapCEU,
    n    = heightDf$N,
    stringsAsFactors = FALSE
  )

  bmiStd <- data.frame(
    rsid = bmiDf$SNP,
    a1   = toupper(bmiDf$A1),
    a2   = toupper(bmiDf$A2),
    beta = bmiDf$b,
    se   = bmiDf$se,
    freq = bmiDf$Freq1.Hapmap,
    n    = bmiDf$N,
    stringsAsFactors = FALSE
  )

  processTraitPair(heightStd, bmiStd, "height", "bmi", pruneStep, mafThreshold)
} else {
  cat("Skipping Height/BMI - files not found in", dataDir, "\n")
}


# Pair 2: SCZ (PGC3, Trubetskoy 2022) / BIP (PGC-BIP 2021, Mullins)
# SCZ VCF columns:  CHROM ID POS A1 A2 FCAS FCON IMPINFO BETA SE PVAL NCAS NCON NEFF
# BIP VCF columns:  1:CHROM 2:POS 3:RSID 4:A1 5:A2 6:BETA 7:SE 8:PVAL
#                   9:NGT 10:FCAS 11:FCON 12:IMPINFO 13:NEFF 14:NCON 15:NTOTAL 16:DIRECTION
sczFile <- file.path(dataDir, "PGC3_SCZ_wave3.european.autosome.public.v3.vcf.tsv")
bipFile <- file.path(dataDir, "pgc-bip2021-all.vcf.tsv")

if (file.exists(sczFile) && file.exists(bipFile)) {
  sczRaw <- read.table(sczFile, header = TRUE, comment.char = "#", stringsAsFactors = FALSE)
  sczStd <- data.frame(
    rsid = sczRaw$ID,
    a1   = toupper(sczRaw$A1),
    a2   = toupper(sczRaw$A2),
    beta = sczRaw$BETA,
    se   = sczRaw$SE,
    freq = sczRaw$FCAS,
    n    = sczRaw$NEFF,
    stringsAsFactors = FALSE
  )

  bipRaw <- read.table(bipFile, header = FALSE, comment.char = "#", stringsAsFactors = FALSE)
  bipStd <- data.frame(
    rsid = bipRaw$V3,
    a1   = toupper(bipRaw$V4),
    a2   = toupper(bipRaw$V5),
    beta = as.numeric(bipRaw$V6),
    se   = as.numeric(bipRaw$V7),
    freq = as.numeric(bipRaw$V10),
    n    = as.numeric(bipRaw$V13),
    stringsAsFactors = FALSE
  )

  processTraitPair(sczStd, bipStd, "scz", "bip", pruneStep, mafThreshold)
} else {
  cat("Skipping SCZ/BIP - files not found in", dataDir, "\n")
}


# Pair 3: HDL / LDL (GLGC Willer 2013)
# Columns: SNP_hg18 SNP_hg19 rsid A1 A2 beta se N P-value Freq.A1.1000G.EUR
hdlFile <- file.path(dataDir, "jointGwasMc_HDL.txt")
ldlFile <- file.path(dataDir, "jointGwasMc_LDL.txt")

if (file.exists(hdlFile) && file.exists(ldlFile)) {
  hdlRaw <- read.table(hdlFile, header = TRUE, stringsAsFactors = FALSE)
  ldlRaw <- read.table(ldlFile, header = TRUE, stringsAsFactors = FALSE)

  hdlStd <- data.frame(
    rsid = hdlRaw$rsid,
    a1   = toupper(hdlRaw$A1),
    a2   = toupper(hdlRaw$A2),
    beta = hdlRaw$beta,
    se   = hdlRaw$se,
    freq = hdlRaw$Freq.A1.1000G.EUR,
    n    = hdlRaw$N,
    stringsAsFactors = FALSE
  )

  ldlStd <- data.frame(
    rsid = ldlRaw$rsid,
    a1   = toupper(ldlRaw$A1),
    a2   = toupper(ldlRaw$A2),
    beta = ldlRaw$beta,
    se   = ldlRaw$se,
    freq = ldlRaw$Freq.A1.1000G.EUR,
    n    = ldlRaw$N,
    stringsAsFactors = FALSE
  )

  processTraitPair(hdlStd, ldlStd, "hdl", "ldl", pruneStep, mafThreshold)
} else {
  cat("Skipping HDL/LDL - files not found in", dataDir, "\n")
}

cat("\nDone. Betahat files written to", dataDir, "\n")
