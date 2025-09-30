'''
Description: 
Author: Chao Ning
Date: 2025-04-03 22:09:33
LastEditTime: 2025-05-12 13:08:46
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

'''
dfG = pd.DataFrame(G)
dfCorr = dfG.corr()
dfCorr.to_csv("LDr.txt", header=False, index=False)

df_lst = []
for i in tqdm(range(10)):
    file = f"/net/zootopia/disk1/chaon/WORK/GxEX/res/21001.10_{i+1}.res"
    df = pd.read_csv(file, sep=r"\s+")
    df_lst.append(df)

df = pd.concat(df_lst)
df.dropna(inplace=True)
dfsid = pd.DataFrame({"SNP": sid_lst})
df = pd.merge(dfsid, df, on="SNP")
beta_lst = [f"beta{i+1}" for i in range(139)]
se_lst = [f"se{i+1}" for i in range(139)]
beta_arr = df.loc[:, beta_lst].values
se_arr = df.loc[:, se_lst].values
z_arr = beta_arr / se_arr
np.savetxt("Z.txt", z_arr)
'''

varcom = [0.272351, 0.220862, 0.253284]
MS.cal_spVi(varcom)
logging.info(f"Vi_sp: {MS.Vi_sp[:5, :5].toarray()}")

res = MS.mmsusie(G, y, L=10)
MS.out_mmsusie(res, "mmsusie_out")

eff = np.sum(res["mu"] * res["alpha"], axis=0)
y = y - G @ eff

prior_cov = MS.create_mixture_prior()

res = MS.mrsusie(G, env_int_arr2, y, prior_cov, estimate_prior_method="optim", maxiter=100, L=2, tol=0.01)
MS.out_mrsusie(res, "mrsusie_out")
