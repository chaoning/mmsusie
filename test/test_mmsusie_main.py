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


# Initialize MMSuSiE model instance
MS = MMSuSiE()

# Define phenotype file path
pheno_file = None

# Read only column names from the phenotype file to determine environmental interactions
pheno_columns = pd.read_csv(pheno_file, sep=r"\s+", nrows=0).columns.tolist()
env_int_names = list()
trait = None

# Read phenotype and environment interaction data into MMSuSiE
MS.read_data(pheno_file, trait=trait, env_int=env_int_names)

# Read sparse GRM for genetic relatedness correction
grm_path = None
MS.read_grm(grm_file=grm_path)

# Extract phenotype (y) and environmental variables (E)
E = MS.get_env_int()
y = MS.get_y(adjust=True)

bedfile = ""
snp_ids = list()
G = MS.get_genotype(bedfile, snp_ids, scale=True)

varcom_file = f"/net/zootopia/disk1/chaon/WORK/GxEX/res/{trait}.var"
varcom = np.loadtxt(varcom_file)
MS.cal_spVi(varcom)

res = MS.mmsusie(G, y, L=10)
out_file = ""
MS.out_mmsusie(res, out_file)
