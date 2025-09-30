'''
Description: 
Author: Chao Ning
Date: 2025-04-03 22:09:33
LastEditTime: 2025-05-02 17:24:04
LastEditors: Chao Ning
'''

import os
# os.environ["NUMEXPR_MAX_THREADS"] = "32"
# os.environ["MKL_NUM_THREADS"] = "32"
# os.environ["OMP_NUM_THREADS"] = "32"
import logging
logging.basicConfig(level=logging.INFO) 
import pandas as pd
import numpy as np
from mmsusie import MMSuSiE
from pysnptools.snpreader import Bed
from tqdm import tqdm


MS = MMSuSiE()
data_file = "/net/zootopia/disk1/chaon/WORK/GxEX/pheno/IINT/pheno.21001.txt"
head_names = pd.read_csv(data_file, sep=r"\s+", nrows=0).columns.tolist()
env_int = head_names[3:]
MS.read_data(data_file, trait="21001", env_int=env_int)

grm_file = "/net/zootopia/disk1/chaon/data/UKB/GRM/ukb_imp"
MS.read_sp_grm(grm_file=grm_file)

env_int_arr2 = MS.get_env_int()
y = MS.get_y(adjust=True)

bedfile = "/net/zootopia/disk1/chaon/data/UKB/imp/ukb_imp_info"
sid_lst = ["rs55872725"]
dfbim = pd.read_csv(bedfile + ".bim", sep=r"\s+", header=None)
dfbim_sub = dfbim[dfbim[1].isin(set(sid_lst))]
chrom = dfbim_sub[0].values[0]
pos = dfbim_sub[3].values[0]
dfbim_sub = dfbim[(dfbim[0] == chrom) & (dfbim[3] > pos - 5000) & (dfbim[3] < pos + 5000)]
sid_lst = dfbim_sub[1].values.tolist()
print("Total SNPs: ", len(sid_lst))

G = MS.get_genotype(bedfile, sid_lst, scale=True)

varcom = [0.272351, 0.220862, 0.253284]
MS.cal_spVi(varcom)
logging.info(f"Vi_sp: {MS.Vi_sp[:5, :5].toarray()}")

res = MS.mmsusie(G, y, L=10)
eff = np.sum(res["mu"] * res["alpha"], axis=0)
y = y - G @ eff

prior_cov = MS.create_mixture_prior()

res = MS.mrsusie(G, env_int_arr2, y, prior_cov, estimate_prior_method="EM", maxiter=100, L=5)
np.savetxt("res_pip_cov_em.txt", res["pip"])
np.savetxt("res_alpha_cov_em.txt", res["alpha"])

with open("res_cs_cov_em.txt", "w") as fin:
    for vec in res["cs"]:
        fin.write(" ".join([str(int(i)) for i in vec]) + "\n")
