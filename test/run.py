'''
Description: 
Author: Chao Ning
Date: 2025-04-03 22:09:33
LastEditTime: 2025-05-02 22:04:39
LastEditors: Chao Ning
'''

import os
import logging
logging.basicConfig(level=logging.INFO) 
import pandas as pd
import numpy as np
from mmsusie import MMSuSiE
from pysnptools.snpreader import Bed
from tqdm import tqdm
import sys

# Get the row index from command-line input
irow = int(sys.argv[1])

# Load the region-level GxE candidate list
region_df = pd.read_csv("/net/zootopia/disk1/chaon/WORK/GxEX/GRL/p_gxe_Identify_Single.csv")
trait = region_df.at[irow, "trait"]
chrom = region_df.at[irow, "chrom"]
start = region_df.at[irow, "start"]
end = region_df.at[irow, "end"]

# Initialize MMSuSiE model instance
MS = MMSuSiE()

# Define phenotype file path
pheno_file = f"/net/zootopia/disk1/chaon/WORK/GxEX/pheno/IINT/pheno.{trait}.txt"

# Read only column names from the phenotype file to determine environmental interactions
pheno_columns = pd.read_csv(pheno_file, sep=r"\s+", nrows=0).columns.tolist()
env_int_names = pheno_columns[3:]  # Assumes first three columns are non-environmental (e.g., ID, age, sex)

# Read phenotype and environment interaction data into MMSuSiE
MS.read_data(pheno_file, trait=trait, env_int=env_int_names)

# Read sparse GRM for genetic relatedness correction
grm_path = "/net/zootopia/disk1/chaon/data/UKB/GRM/ukb_imp"
MS.read_sp_grm(grm_file=grm_path)

# Extract phenotype (y) and environmental variables (E)
E = MS.get_env_int()
y = MS.get_y(adjust=True)


# Load BIM file to extract SNPs located within the target genomic region
bedfile = "/net/zootopia/disk1/chaon/data/UKB/imp/ukb_imp_info"
bim_file = bedfile + ".bim"
bim_df = pd.read_csv(bim_file, sep=r"\s+", header=None, usecols=[0, 1, 3], names=["chrom", "snp_id", "pos"])

# Filter SNPs that fall within the specified chromosome and base pair range
snp_in_region = bim_df.query("(chrom == @chrom) and (pos > @start) and (pos < @end)")
snp_ids = snp_in_region["snp_id"].tolist()

# Print region and SNP summary
print(f"Trait: {trait}")
print(f"Chromosome: {chrom}, Region: {start}-{end}")
print(f"Total SNPs in region: {len(snp_ids)}")

G = MS.get_genotype(bedfile, snp_ids, scale=True)

varcom_file = f"/net/zootopia/disk1/chaon/WORK/GxEX/res/{trait}.var"
varcom = np.loadtxt(varcom_file)
MS.cal_spVi(varcom)

res = MS.mmsusie(G, y, L=10)
MS.out_mmsusie(res, f"/net/zootopia/disk1/chaon/WORK/GxEX/mrsusie/res/{irow}.mmsusie_out")

eff = np.sum(res["mu"] * res["alpha"], axis=0)
y = y - G @ eff

prior_cov = MS.create_mixture_prior()

res = MS.mrsusie(G, E, y, prior_cov, estimate_prior_method="optim", maxiter=100, L=10)
MS.out_mrsusie(res, f"/net/zootopia/disk1/chaon/WORK/GxEX/mrsusie/res/{irow}.mrsusie_out")
