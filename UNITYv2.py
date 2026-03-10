#!/usr/bin/env python3
"""
UNITY v2 - Reimplementation of Johnson et al. (2018)
  Extension 1: Data-augmented partial Gibbs sampler
  Extension 2: Sample overlap correction via rho_0 parameter
  Adaptive proposal scaling targeting ~25% acceptance

Usage:
    python unity_v2.py --sim \
        --A00 0.95 --A10 0.02 --A01 0.02 --A11 0.01 \
        --N1 100000 --N2 100000 --H1 0.05 --H2 0.05 --rho 0.25 \
        --M 100000 --ITS 2000 --method gibbs --id sim_gibbs

    python unity_v2.py --file1 height.txt --file2 bmi.txt \
        --N1 253213 --N2 339224 --H1 0.239 --H2 0.157 --rho 0.085 \
        --M 214921 --ITS 2000 --method mh --id height_bmi
"""

import numpy as np
import json
import argparse
import sys
import time
from scipy.special import logsumexp
from scipy.stats import dirichlet, uniform, norm


# Global constants
LOG_TINY = 1e-308 # To avoid log(p) = log(0)
DIRICHLET_PRIOR = 0.2 # Dirichlet concentration; matches original UNITY (lam1-4 = 0.2)
DEFAULT_SEED = 486
DEFAULT_ITS = 2000
DEFAULT_BURN_FRAC = 0.25
RHO0_PROPOSAL_SD = 0.02 # step size for rho0 random-walk proposal

# Adaptive proposal: start with B_INIT, adjust every ADAPT_INTERVAL iterations
# targeting ADAPT_TARGET acceptance rate
# NOTE: original UNITY used a fixed B = 10 Dirichlet proposal (UNITY_ismb.py, run_mcmc()), 
#       We initialize at B = 100 and adapt during burn-in only
B_INIT = 100
ADAPT_INTERVAL = 50
ADAPT_TARGET = 0.25
ADAPT_FACTOR = 1.5

LOG_2PI = np.log(2.0 * np.pi)

# Vectorized log-pdf for a zero-mean 2x2 bivariate normal distribution
# Operates on arrays x1, x2 of length M (one entry per SNP) and scalar
# variance/covariance parameters
# Returns an array of log-pdf values
# NOTE: replaces per-SNP calls to scipy.stats.multivariate_normal.pdf used
#   in the original UNITY (UNITY_ismb.py, neg_log_p_pdf_fast_pvec)
#   this avoids per-call overhead and gives roughly a 7x speedup on M > 100K
def logpdfBivariateNorm(x1, x2, var1, var2, cov12=0.0):
    det = var1 * var2 - cov12 * cov12
    if det <= 0:
        return np.full_like(x1, -np.inf)
    logDet = np.log(det)
    invDet = 1.0 / det
    mahal = invDet * (var2 * x1 * x1 - 2.0 * cov12 * x1 * x2 + var1 * x2 * x2)
    return -0.5 * (2.0 * LOG_2PI + logDet + mahal)


# Compute per-component covariance parameters from the model's hyperparameters
# and the current proportion vector
# Inputs: proportions (p00, p10, p01, p11), SNP heritabilities h2trait1/h2trait2,
#   genetic correlation geneticCorr, sample sizes sampleSize1/sampleSize2,
#   total SNP count numSnps, and optional sample overlap parameter rho0
# Returns a dict with noise variances, causal effect variances, and a PSD flag
# NOTE: adapted from the covariance construction in UNITY_ismb.py (neg_log_p_pdf_fast_pvec)
#   Variable names updated: sig_gam1 -> sigGamma1, sigma_beta -> noiseVar, cov_e -> covEnv
#   The original used nearPSD() to patch non-PSD matrices,
#   we return isPSD=False instead and let the caller handle it
def computeCovarianceParams(proportions, h2trait1, h2trait2, geneticCorr,
                            sampleSize1, sampleSize2, numSnps, rho0=0.0):
    p00, p10, p01, p11 = proportions
    noiseVar1 = (1.0 - h2trait1) / sampleSize1
    noiseVar2 = (1.0 - h2trait2) / sampleSize2
    covEnv = rho0 * np.sqrt(noiseVar1 * noiseVar2)
    sigGamma1 = h2trait1 / (numSnps * (p11 + p10))
    sigGamma2 = h2trait2 / (numSnps * (p11 + p01))
    sigGammaX = (np.sqrt(h2trait1) * np.sqrt(h2trait2) * geneticCorr) / (numSnps * p11)
    var11_1 = sigGamma1 + noiseVar1
    var11_2 = sigGamma2 + noiseVar2
    cov11_offdiag = sigGammaX + covEnv
    det11 = var11_1 * var11_2 - cov11_offdiag * cov11_offdiag
    return {
        "noiseVar1": noiseVar1, "noiseVar2": noiseVar2, "covEnv": covEnv,
        "sigGamma1": sigGamma1, "sigGamma2": sigGamma2,
        "sigGammaX": sigGammaX, "isPSD": det11 > 0,
    }


# Evaluate the log-pdf of each SNP under each of the four mixture components
# Takes the output of computeCovarianceParams and arrays betaHat1/betaHat2
# Returns a (4, M) array: rows index the component (00, 10, 01, 11)
def computeComponentLogPdfs(betaHat1, betaHat2, covParams):
    nv1 = covParams["noiseVar1"]; nv2 = covParams["noiseVar2"]
    ce = covParams["covEnv"]; sg1 = covParams["sigGamma1"]
    sg2 = covParams["sigGamma2"]; sgx = covParams["sigGammaX"]
    logPdf00 = logpdfBivariateNorm(betaHat1, betaHat2, nv1, nv2, ce)
    logPdf10 = logpdfBivariateNorm(betaHat1, betaHat2, sg1 + nv1, nv2, ce)
    logPdf01 = logpdfBivariateNorm(betaHat1, betaHat2, nv1, sg2 + nv2, ce)
    logPdf11 = logpdfBivariateNorm(betaHat1, betaHat2,
                                    sg1 + nv1, sg2 + nv2, sgx + ce)
    return np.stack([logPdf00, logPdf10, logPdf01, logPdf11], axis=0)


# Marginal log-likelihood summed over all SNPs, integrating out the latent
# per-SNP class assignments + Returns a scalar
# NOTE: adapted from neg_log_p_pdf_fast_pvec in UNITY_ismb.py. Uses logsumexp
#   throughout for numerical stability; original operated in probability space
#   with explicit zero-replacement hacks
def computeLogLikelihood(betaHat1, betaHat2, proportions, h2trait1, h2trait2,
                         geneticCorr, sampleSize1, sampleSize2, numSnps, rho0=0.0):
    covParams = computeCovarianceParams(
        proportions, h2trait1, h2trait2, geneticCorr,
        sampleSize1, sampleSize2, numSnps, rho0)
    if not covParams["isPSD"]:
        return -np.inf
    componentLogPdfs = computeComponentLogPdfs(betaHat1, betaHat2, covParams)
    logWeights = np.log(np.maximum(proportions, LOG_TINY))
    logWeighted = componentLogPdfs + logWeights[:, np.newaxis]
    return np.sum(logsumexp(logWeighted, axis=0))


# Log-posterior: marginal log-likelihood plus the Dirichlet log-prior on p
# The Dirichlet prior uses concentration kappa = 0.2, matching the original
# UNITY (lam1 = lam2 = lam3 = lam4 = 0.2 in UNITY_ismb.py).
def computeLogPosterior(betaHat1, betaHat2, proportions, h2trait1, h2trait2,
                        geneticCorr, sampleSize1, sampleSize2, numSnps, rho0=0.0):
    logLik = computeLogLikelihood(
        betaHat1, betaHat2, proportions, h2trait1, h2trait2,
        geneticCorr, sampleSize1, sampleSize2, numSnps, rho0)
    if logLik == -np.inf:
        return -np.inf
    return logLik + dirichlet.logpdf(proportions, [DIRICHLET_PRIOR] * 4)


# Conditional log-likelihood given fixed per-SNP class assignments
# Used in Block 2 of the Gibbs sampler, classAssignments is a length-M
# integer array with values in {0, 1, 2, 3}
def computeLogLikGivenAssignments(betaHat1, betaHat2, classAssignments,
                                   covParams, numSnps):
    if not covParams["isPSD"]:
        return -np.inf
    componentLogPdfs = computeComponentLogPdfs(betaHat1, betaHat2, covParams)
    return np.sum(componentLogPdfs[classAssignments, np.arange(numSnps)])


# Dirichlet random-walk proposal: Dir(kappa + B * p_current).
# Returns the proposed p vector and the forward alpha vecto
# NOTE: the original UNITY (UNITY_ismb.py, run_mcmc) used a fixed B = 10,
#   Here B is adaptive (see adaptB)
def proposeDirichlet(pCurrent, B):
    alphaVec = np.maximum(DIRICHLET_PRIOR + B * pCurrent, 0.1)
    pStar = dirichlet.rvs(alphaVec).ravel()
    return pStar, alphaVec


# Log proposal ratio for the asymmetric MH correction
# Computes log q(p_current | p_star) - log q(p_star | p_current)
# This is needed because the Dirichlet proposal is not symmetric
# NOTE: the original UNITY did not include this correction
def logProposalRatio(pStar, pCurrent, B):
    alphaForward = np.maximum(DIRICHLET_PRIOR + B * pCurrent, 0.1)
    alphaReverse = np.maximum(DIRICHLET_PRIOR + B * pStar, 0.1)
    return dirichlet.logpdf(pCurrent, alphaReverse) - dirichlet.logpdf(pStar, alphaForward)


# Adjust proposal concentration B to push acceptance toward ~25%
# B is increased (tighter proposal, more local steps) when acceptance is too low,
# and decreased (wider proposal, more exploration) when it is too high
# Called every ADAPT_INTERVAL iterations during burn-in only
def adaptB(B, recentAcceptRate):
    if recentAcceptRate < 0.15:
        return B * ADAPT_FACTOR
    elif recentAcceptRate > 0.35:
        return B / ADAPT_FACTOR
    return B


# Simulate GWAS summary statistics under the UNITY generative model
# Draws SNP classes from Multinomial(p), causal effects from a bivariate
# normal, and adds correlated Gaussian noise scaled by (1 - h2) / N
# rho0Sim adds correlated noise to model sample overlap
# Returns arrays betaHat1, betaHat2 of length numSnps
# NOTE: adapted from simulate() in UNITY_ismb.py, Vectorized using NumPy
#   multinomial and multivariate_normal rather than the original per-SNP loop
def simulateSummaryStats(numSnps, sampleSize1, sampleSize2,
                         h2trait1, h2trait2, geneticCorr,
                         propNull, propTrait1, propTrait2, propShared,
                         rho0Sim=0.0):
    varCausal1 = h2trait1 / (numSnps * (propShared + propTrait1))
    varCausal2 = h2trait2 / (numSnps * (propShared + propTrait2))
    covCausal = (np.sqrt(h2trait1) * np.sqrt(h2trait2) * geneticCorr) / \
                (numSnps * propShared)
    maxCov = np.sqrt(varCausal1 * varCausal2)
    if abs(covCausal) >= maxCov:
        covCausalClamped = np.sign(covCausal) * maxCov * 0.99
        print(f"WARNING: Clamped covCausal from {covCausal:.4e} to {covCausalClamped:.4e}")
        covCausal = covCausalClamped
    covMatrix = np.array([[varCausal1, covCausal], [covCausal, varCausal2]])
    classProbs = [propNull, propTrait1, propTrait2, propShared]
    snpClasses = np.random.multinomial(1, classProbs, size=numSnps)
    allEffects = np.random.multivariate_normal([0, 0], covMatrix, size=numSnps)
    maskTrait1 = snpClasses[:, 1] + snpClasses[:, 3]
    maskTrait2 = snpClasses[:, 2] + snpClasses[:, 3]
    trueBeta1 = allEffects[:, 0] * maskTrait1
    trueBeta2 = allEffects[:, 1] * maskTrait2
    noiseVar1 = (1.0 - h2trait1) / sampleSize1
    noiseVar2 = (1.0 - h2trait2) / sampleSize2
    covEnv = rho0Sim * np.sqrt(noiseVar1 * noiseVar2)
    noiseCov = np.array([[noiseVar1, covEnv], [covEnv, noiseVar2]])
    noise = np.random.multivariate_normal([0, 0], noiseCov, size=numSnps)
    betaHat1 = trueBeta1 + noise[:, 0]
    betaHat2 = trueBeta2 + noise[:, 1]
    counts = snpClasses.sum(axis=0)
    print(f"Simulated SNP counts: null={counts[0]}, trait1={counts[1]}, "
          f"trait2={counts[2]}, shared={counts[3]}")
    return betaHat1, betaHat2


# Load preprocessed effect size files for two traits and check for scale issues.
# filePath1/filePath2 contain one betahat per line (single-column) 
# sampleSize1/h2trait1 are used only for the scale-mismatch heuristic, not for inference
# NOTE: original UNITY's load_sumstats() called np.loadtxt with no validation,
#   Added scale detection to catch cases where Z-scores are passed instead of effect sizes
def loadSummaryStats(filePath1, filePath2, betaCol=None,
                     sampleSize1=None, sampleSize2=None,
                     h2trait1=None, h2trait2=None):
    betaHat1 = _loadFlexible(filePath1, betaCol)
    betaHat2 = _loadFlexible(filePath2, betaCol)
    if len(betaHat1) != len(betaHat2):
        raise ValueError(f"Mismatched SNP counts: {len(betaHat1)} vs {len(betaHat2)}")
    print(f"Loaded {len(betaHat1)} SNPs from summary statistics files.")
    if sampleSize1 is not None and h2trait1 is not None:
        expectedSd = np.sqrt((1.0 - h2trait1) / sampleSize1)
        observedSd = np.std(betaHat1)
        ratio = observedSd / expectedSd
        if ratio > 10:
            print(f"\n{'!'*60}")
            print(f"WARNING: Data scale mismatch! Expected sd ~ {expectedSd:.4e}, "
                  f"observed {observedSd:.4e} ({ratio:.0f}x too large)")
            print(f"{'!'*60}\n")
    return betaHat1, betaHat2


def _loadFlexible(filePath, betaCol=None):
    try:
        data = np.loadtxt(filePath)
    except ValueError:
        data = np.loadtxt(filePath, skiprows=1)
    if data.ndim == 1:
        return data
    col = betaCol if betaCol is not None else data.shape[1] - 1
    print(f"  Multi-column file ({data.shape[1]} cols), using column {col}")
    return data[:, col]


# Sample per-SNP class assignments c_m | p, z for all M SNPs in one vectorized pass.
# Each assignment is drawn from a categorical distribution with probabilities
# proportional to p_k * N(z_m; 0, Sigma_k). This is Block 1 of the Gibbs sampler.
# Returns classAssignments (length-M int array in {0,1,2,3}) and classCounts (length-4).
def sampleClassAssignments(betaHat1, betaHat2, proportions, covParams, numSnps):
    componentLogPdfs = computeComponentLogPdfs(betaHat1, betaHat2, covParams)
    logWeights = np.log(np.maximum(proportions, LOG_TINY))
    logResp = componentLogPdfs + logWeights[:, np.newaxis]
    logNorm = logsumexp(logResp, axis=0, keepdims=True)
    resp = np.exp(logResp - logNorm).T  # (M, 4)
    cumProbs = np.cumsum(resp, axis=1)
    u = np.random.uniform(size=(numSnps, 1))
    classAssignments = (u > cumProbs).sum(axis=1).astype(int)
    classAssignments = np.clip(classAssignments, 0, 3)
    classCounts = np.bincount(classAssignments, minlength=4)
    return classAssignments, classCounts


# Adaptive Metropolis-Hastings sampler using the marginal likelihood
# (class assignments integrated out). This is the primary sampler for
# real GWAS analyses.
# rho0 can be fixed (passed as a scalar) or jointly inferred (--infer_rho0),
# in which case a symmetric normal random-walk step is added each iteration.
# Returns a results dict; see _buildResults for structure.
# NOTE: adapted from run_mcmc() in UNITY_ismb.py. Key changes:
#   adaptive B replacing fixed B = 10; asymmetric proposal correction added;
#   log-space throughout; no nearPSD patching.
def runMH(betaHat1, betaHat2, h2trait1, h2trait2, geneticCorr,
          sampleSize1, sampleSize2, numSnps, numIterations,
          rho0=0.0, inferRho0=False, burnFraction=DEFAULT_BURN_FRAC):

    numBurnin = int(numIterations * burnFraction)

    pCurrent = dirichlet.rvs([DIRICHLET_PRIOR] * 4).ravel()
    rho0Current = rho0
    logPostCurrent = computeLogPosterior(
        betaHat1, betaHat2, pCurrent, h2trait1, h2trait2,
        geneticCorr, sampleSize1, sampleSize2, numSnps, rho0Current)

    chainP = np.zeros((numIterations, 4))
    chainRho0 = np.zeros(numIterations)
    numAccepted = 0; numAcceptedRho0 = 0
    recentAccepts = 0
    B = float(B_INIT)
    startTime = time.time()

    for iteration in range(numIterations):
        # Propose p
        pStar, _ = proposeDirichlet(pCurrent, B)
        logPostStar = computeLogPosterior(
            betaHat1, betaHat2, pStar, h2trait1, h2trait2,
            geneticCorr, sampleSize1, sampleSize2, numSnps, rho0Current)
        logMHratio = (logPostStar - logPostCurrent) + logProposalRatio(pStar, pCurrent, B)

        if np.log(uniform.rvs()) < logMHratio:
            pCurrent = pStar; logPostCurrent = logPostStar
            numAccepted += 1; recentAccepts += 1

        # Propose rho0 if inferring
        if inferRho0:
            rho0Star = np.clip(norm.rvs(loc=rho0Current, scale=RHO0_PROPOSAL_SD), -0.99, 0.99)
            logPostR = computeLogPosterior(
                betaHat1, betaHat2, pCurrent, h2trait1, h2trait2,
                geneticCorr, sampleSize1, sampleSize2, numSnps, rho0Star)
            if np.log(uniform.rvs()) < (logPostR - logPostCurrent):
                rho0Current = rho0Star; logPostCurrent = logPostR; numAcceptedRho0 += 1

        chainP[iteration] = pCurrent; chainRho0[iteration] = rho0Current

        # Adapt B during burn-in only
        if (iteration + 1) % ADAPT_INTERVAL == 0 and iteration < numBurnin:
            rate = recentAccepts / ADAPT_INTERVAL
            B = adaptB(B, rate)
            recentAccepts = 0

        if iteration % 200 == 0:
            elapsed = time.time() - startTime
            rate = (iteration + 1) / elapsed if elapsed > 0 else 0
            rStr = f" rho0={rho0Current:.4f}" if inferRho0 or rho0 != 0 else ""
            print(f"  [MH] iter {iteration:5d}/{numIterations} | "
                  f"p00={pCurrent[0]:.4f} p10={pCurrent[1]:.4f} "
                  f"p01={pCurrent[2]:.4f} p11={pCurrent[3]:.4f}{rStr} | "
                  f"B={B:.0f} accept={numAccepted/(iteration+1):.2%} | {rate:.1f} it/s")

    return _buildResults(chainP, chainRho0, numAccepted, numAcceptedRho0,
                         numIterations, numBurnin, startTime, inferRho0, B)


# Data-augmented partial Gibbs sampler
# Each iteration has three blocks:
#   Block 1 (exact): sample c_m | p, z for all SNPs
#   Block 2 (MH): update p using the conditional likelihood given assignments
#   Block 3 (optional MH): update rho0 if inferring sample overlap
# Achieves higher acceptance rates on simulated data but exhibits a collapsing
# bias on real GWAS data - see project writeup for the identifiability analysis.
# Returns a results dict with an extra "chainCounts" key.
def runGibbs(betaHat1, betaHat2, h2trait1, h2trait2, geneticCorr,
             sampleSize1, sampleSize2, numSnps, numIterations,
             rho0=0.0, inferRho0=False, burnFraction=DEFAULT_BURN_FRAC):

    numBurnin = int(numIterations * burnFraction)

    pCurrent = dirichlet.rvs([DIRICHLET_PRIOR] * 4).ravel()
    rho0Current = rho0

    chainP = np.zeros((numIterations, 4))
    chainCounts = np.zeros((numIterations, 4))
    chainRho0 = np.zeros(numIterations)
    numAcceptedP = 0; numAcceptedRho0 = 0
    recentAccepts = 0
    B = float(B_INIT)
    startTime = time.time()

    for iteration in range(numIterations):
        # Block 1: sample class assignments c | p, z
        covParams = computeCovarianceParams(
            pCurrent, h2trait1, h2trait2, geneticCorr,
            sampleSize1, sampleSize2, numSnps, rho0Current)

        if not covParams["isPSD"]:
            pCurrent = dirichlet.rvs([DIRICHLET_PRIOR] * 4).ravel()
            chainP[iteration] = pCurrent; chainRho0[iteration] = rho0Current
            continue

        classAssignments, classCounts = sampleClassAssignments(
            betaHat1, betaHat2, pCurrent, covParams, numSnps)

        # Block 2: MH update for p given class assignments
        pStar, _ = proposeDirichlet(pCurrent, B)
        covParamsStar = computeCovarianceParams(
            pStar, h2trait1, h2trait2, geneticCorr,
            sampleSize1, sampleSize2, numSnps, rho0Current)
        logLikStar = computeLogLikGivenAssignments(
            betaHat1, betaHat2, classAssignments, covParamsStar, numSnps)
        logLikCurr = computeLogLikGivenAssignments(
            betaHat1, betaHat2, classAssignments, covParams, numSnps)

        logMH = ((logLikStar + dirichlet.logpdf(pStar, [DIRICHLET_PRIOR]*4)) -
                 (logLikCurr + dirichlet.logpdf(pCurrent, [DIRICHLET_PRIOR]*4))) + \
                logProposalRatio(pStar, pCurrent, B)

        if np.log(uniform.rvs()) < logMH:
            pCurrent = pStar; numAcceptedP += 1; recentAccepts += 1

        # Block 3: MH update for rho0
        if inferRho0:
            rho0Star = np.clip(norm.rvs(loc=rho0Current, scale=RHO0_PROPOSAL_SD), -0.99, 0.99)
            covR = computeCovarianceParams(pCurrent, h2trait1, h2trait2, geneticCorr,
                                           sampleSize1, sampleSize2, numSnps, rho0Star)
            covC = computeCovarianceParams(pCurrent, h2trait1, h2trait2, geneticCorr,
                                           sampleSize1, sampleSize2, numSnps, rho0Current)
            logLR = computeLogLikGivenAssignments(betaHat1, betaHat2, classAssignments, covR, numSnps)
            logLC = computeLogLikGivenAssignments(betaHat1, betaHat2, classAssignments, covC, numSnps)
            logMHr = (logLR + norm.logpdf(rho0Star, 0, 0.5)) - \
                     (logLC + norm.logpdf(rho0Current, 0, 0.5))
            if np.log(uniform.rvs()) < logMHr:
                rho0Current = rho0Star; numAcceptedRho0 += 1

        chainP[iteration] = pCurrent
        chainCounts[iteration] = classCounts
        chainRho0[iteration] = rho0Current

        # Adapt B during burn-in only
        if (iteration + 1) % ADAPT_INTERVAL == 0 and iteration < numBurnin:
            rate = recentAccepts / ADAPT_INTERVAL
            B = adaptB(B, rate)
            recentAccepts = 0

        if iteration % 200 == 0:
            elapsed = time.time() - startTime
            rate = (iteration + 1) / elapsed if elapsed > 0 else 0
            rStr = f" rho0={rho0Current:.4f}" if inferRho0 or rho0 != 0 else ""
            print(f"  [Gibbs] iter {iteration:5d}/{numIterations} | "
                  f"p00={pCurrent[0]:.4f} p10={pCurrent[1]:.4f} "
                  f"p01={pCurrent[2]:.4f} p11={pCurrent[3]:.4f}{rStr} | "
                  f"counts=[{classCounts[0]:.0f},{classCounts[1]:.0f},"
                  f"{classCounts[2]:.0f},{classCounts[3]:.0f}] | "
                  f"B={B:.0f} accept={numAcceptedP/(iteration+1):.2%} | {rate:.1f} it/s")

    results = _buildResults(chainP, chainRho0, numAcceptedP, numAcceptedRho0,
                            numIterations, numBurnin, startTime, inferRho0, B)
    results["chainCounts"] = chainCounts
    return results


# Compile posterior summaries and diagnostics from a completed chain.
# Post-burnin samples are used for all estimates. rho0 summaries are only
# populated when inferRho0 is True or rho0 was fixed to a nonzero value.
def _buildResults(chainP, chainRho0, numAcceptedP, numAcceptedRho0,
                  numIterations, numBurnin, startTime, inferRho0, finalB):
    elapsed = time.time() - startTime
    postBurninChain = chainP[numBurnin:]
    posteriorMeans = postBurninChain.mean(axis=0)
    posteriorStds = postBurninChain.std(axis=0)
    results = {
        "posteriorMeans": posteriorMeans,
        "posteriorStds": posteriorStds,
        "acceptRate": numAcceptedP / numIterations,
        "chain": chainP,
        "chainRho0": chainRho0,
        "elapsedSeconds": elapsed,
        "numBurnin": numBurnin,
        "numIterations": numIterations,
        "finalB": finalB,
    }
    if inferRho0:
        postBurninRho0 = chainRho0[numBurnin:]
        results["rho0Mean"] = postBurninRho0.mean()
        results["rho0Std"] = postBurninRho0.std()
        results["rho0AcceptRate"] = numAcceptedRho0 / numIterations
    else:
        results["rho0Mean"] = chainRho0[0] if len(chainRho0) > 0 else 0.0
        results["rho0Std"] = 0.0
    return results


# Print posterior estimates to stdout and optionally write three output files:
#   out.{id}.{seed}.txt - text summary
#   out.{id}.{seed}_chain.txt - full MCMC chain (p00 p10 p01 p11 rho0)
#   out.{id}.{seed}.json - clean results + config + diagnostics
# trueProportions and trueRho0 are only passed for simulation runs
def printResults(results, trueProportions=None, trueRho0=None,
                 outputFile=None, method="", config=None):
    means = results["posteriorMeans"]
    stds = results["posteriorStds"]

    lines = []
    lines.append("=" * 60)
    lines.append(f"UNITY v2 Results - method: {method}")
    lines.append("=" * 60)
    lines.append(f"  Acceptance rate (p): {results['acceptRate']:.2%}")
    lines.append(f"  Final proposal B:    {results['finalB']:.0f}")
    lines.append(f"  Runtime:             {results['elapsedSeconds']:.1f}s")
    lines.append(f"  Burn-in:             {results['numBurnin']}")
    lines.append(f"  Post-burnin samples: {results['numIterations'] - results['numBurnin']}")
    lines.append("")
    lines.append("  Parameter   Estimate      Std")
    lines.append("  ---------   --------      ---")
    paramNames = ["p00", "p10", "p01", "p11"]
    for i, name in enumerate(paramNames):
        line = f"  {name:10s}  {means[i]:.6f}    {stds[i]:.6f}"
        if trueProportions is not None:
            line += f"    (true: {trueProportions[i]:.4f})"
        lines.append(line)

    rho0Mean = results.get("rho0Mean", 0.0)
    rho0Std = results.get("rho0Std", 0.0)
    if rho0Std > 0 or rho0Mean != 0:
        line = f"  {'rho0':10s}  {rho0Mean:.6f}    {rho0Std:.6f}"
        if trueRho0 is not None:
            line += f"    (true: {trueRho0:.4f})"
        lines.append(line)
        if "rho0AcceptRate" in results:
            lines.append(f"  rho0 accept rate: {results['rho0AcceptRate']:.2%}")
    lines.append("=" * 60)

    output = "\n".join(lines)
    print(output)

    if outputFile is not None:
        with open(outputFile, 'w') as f:
            f.write(output + "\n")
        chainFile = outputFile.replace(".txt", "_chain.txt")
        chainData = np.column_stack([results["chain"], results["chainRho0"]])
        np.savetxt(chainFile, chainData,
                   header="p00 p10 p01 p11 rho0", fmt="%.8f")
        jsonFile = outputFile.replace(".txt", ".json")
        jsonOut = {
            "method": method,
            "config": config or {},
            "results": {
                "p00": {"mean": float(means[0]), "std": float(stds[0])},
                "p10": {"mean": float(means[1]), "std": float(stds[1])},
                "p01": {"mean": float(means[2]), "std": float(stds[2])},
                "p11": {"mean": float(means[3]), "std": float(stds[3])},
                "rho0": {"mean": float(rho0Mean), "std": float(rho0Std)},
            },
            "diagnostics": {
                "acceptRate": results["acceptRate"],
                "finalB": results["finalB"],
                "runtime": results["elapsedSeconds"],
                "burnin": results["numBurnin"],
                "numIterations": results["numIterations"],
                "postBurninSamples": results["numIterations"] - results["numBurnin"],
            },
        }
        if trueProportions is not None:
            jsonOut["trueValues"] = {
                "p00": trueProportions[0], "p10": trueProportions[1],
                "p01": trueProportions[2], "p11": trueProportions[3],
            }
            if trueRho0 is not None and trueRho0 != 0:
                jsonOut["trueValues"]["rho0"] = trueRho0
        with open(jsonFile, 'w') as f:
            json.dump(jsonOut, f, indent=2)
        print(f"Chain saved to {chainFile}")
        print(f"JSON saved to {jsonFile}")


def parseArguments():
    parser = argparse.ArgumentParser(
        description="UNITY v2: Estimate shared/trait-specific causal proportions")
    parser.add_argument("--sim", action="store_true",
                        help="Simulate summary statistics instead of loading files")
    parser.add_argument("--file1", type=str,
                        help="Effect size file for trait 1 (one betahat per line)")
    parser.add_argument("--file2", type=str,
                        help="Effect size file for trait 2")
    parser.add_argument("--beta_col", type=int, default=None,
                        help="Column index for beta in multi-column files (0-indexed)")
    parser.add_argument("--H1", type=float, required=True, help="SNP heritability, trait 1")
    parser.add_argument("--H2", type=float, required=True, help="SNP heritability, trait 2")
    parser.add_argument("--rho", type=float, default=0.0, help="Genetic correlation")
    parser.add_argument("--N1", type=int, required=True, help="Sample size, trait 1")
    parser.add_argument("--N2", type=int, required=True, help="Sample size, trait 2")
    parser.add_argument("--M", type=int, required=True, help="Number of SNPs")
    parser.add_argument("--rho0", type=float, default=0.0,
                        help="Fixed sample overlap parameter (0 = no overlap)")
    parser.add_argument("--infer_rho0", action="store_true",
                        help="Jointly infer rho0 within the MCMC")
    parser.add_argument("--rho0_sim", type=float, default=0.0,
                        help="Simulate data with this level of sample overlap")
    parser.add_argument("--A00", type=float, help="True p00 for simulation")
    parser.add_argument("--A10", type=float, help="True p10 for simulation")
    parser.add_argument("--A01", type=float, help="True p01 for simulation")
    parser.add_argument("--A11", type=float, help="True p11 for simulation")
    parser.add_argument("--ITS", type=int, default=DEFAULT_ITS, help="Total MCMC iterations")
    parser.add_argument("--burn", type=float, default=DEFAULT_BURN_FRAC,
                        help="Burn-in fraction (default 0.25)")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed")
    parser.add_argument("--id", type=str, default="unity_run",
                        help="Output file identifier")
    parser.add_argument("--method", type=str, default="gibbs",
                        choices=["mh", "gibbs"],
                        help="Sampler: 'mh' (adaptive MH) or 'gibbs' (data-augmented)")
    return parser.parse_args()

# Set seed for reproducible results
# Added error calls/warnings to aid in re-running
def main():
    args = parseArguments()
    np.random.seed(args.seed)
    trueProportions = None; trueRho0 = None

    if args.sim:
        if None in [args.A00, args.A10, args.A01, args.A11]:
            print("ERROR: --sim requires --A00, --A10, --A01, --A11"); sys.exit(1)
        pSum = args.A00 + args.A10 + args.A01 + args.A11
        if abs(pSum - 1.0) > 1e-6:
            print(f"ERROR: proportions sum to {pSum}"); sys.exit(1)
        trueProportions = [args.A00, args.A10, args.A01, args.A11]
        trueRho0 = args.rho0_sim
        print("Simulating summary statistics...")
        betaHat1, betaHat2 = simulateSummaryStats(
            numSnps=args.M, sampleSize1=args.N1, sampleSize2=args.N2,
            h2trait1=args.H1, h2trait2=args.H2, geneticCorr=args.rho,
            propNull=args.A00, propTrait1=args.A10,
            propTrait2=args.A01, propShared=args.A11, rho0Sim=args.rho0_sim)
    else:
        if args.file1 is None or args.file2 is None:
            print("ERROR: provide --file1 and --file2, or use --sim"); sys.exit(1)
        betaHat1, betaHat2 = loadSummaryStats(
            args.file1, args.file2, betaCol=args.beta_col,
            sampleSize1=args.N1, sampleSize2=args.N2,
            h2trait1=args.H1, h2trait2=args.H2)

    actualM = len(betaHat1)
    if actualM != args.M:
        print(f"WARNING: --M={args.M} but loaded {actualM} SNPs. Using {actualM}.")
        args.M = actualM

    rho0 = args.rho0
    config = {
        "method": args.method, "M": args.M, "N1": args.N1, "N2": args.N2,
        "H1": args.H1, "H2": args.H2, "rho": args.rho, "rho0": rho0,
        "inferRho0": args.infer_rho0, "ITS": args.ITS, "seed": args.seed,
        "burnFraction": args.burn, "id": args.id,
    }

    print(f"\n{'='*60}")
    print(f"UNITY v2 - Configuration")
    print(f"{'='*60}")
    print(f"  Method:    {args.method.upper()}")
    print(f"  SNPs:      {args.M}")
    print(f"  N1={args.N1}, N2={args.N2}")
    print(f"  h2_1={args.H1}, h2_2={args.H2}, rho={args.rho}")
    if rho0 != 0 or args.infer_rho0:
        print(f"  rho0={rho0} (infer={args.infer_rho0})")
    print(f"  Iterations: {args.ITS} (burnin: {int(args.ITS * args.burn)})")
    if trueProportions is not None:
        print(f"  True p: {trueProportions}")
    print(f"{'='*60}\n")

    if args.method == "mh":
        print("Running MH sampler...")
        results = runMH(betaHat1, betaHat2, h2trait1=args.H1, h2trait2=args.H2,
                        geneticCorr=args.rho, sampleSize1=args.N1, sampleSize2=args.N2,
                        numSnps=args.M, numIterations=args.ITS,
                        rho0=rho0, inferRho0=args.infer_rho0, burnFraction=args.burn)
    else:
        print("Running data-augmented Gibbs sampler...")
        results = runGibbs(betaHat1, betaHat2, h2trait1=args.H1, h2trait2=args.H2,
                           geneticCorr=args.rho, sampleSize1=args.N1, sampleSize2=args.N2,
                           numSnps=args.M, numIterations=args.ITS,
                           rho0=rho0, inferRho0=args.infer_rho0, burnFraction=args.burn)

    outputFile = f"out.{args.id}.{args.seed}.txt"
    printResults(results, trueProportions=trueProportions, trueRho0=trueRho0,
                 outputFile=outputFile, method=args.method, config=config)


if __name__ == "__main__":
    main()
