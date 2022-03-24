# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2022/3/11 09:42
"""
import re

import numpy as np
import pandas as pd
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
from statsmodels.stats.multicomp import pairwise_tukeyhsd

from ORCSRS.Config import experiments_replication_csv


def anova_var(X_vars: list, Y_var, df):
    formula = f'{Y_var}~ {" + ".join(X_vars)}'
    model = ols(formula, df).fit()
    anova_res = anova_lm(model)
    return anova_res


df = pd.read_csv(experiments_replication_csv, engine='python', index_col=False)

col_mapping = {i: re.sub(r"[{}]|\(.*\)", '', re.sub(r'\s', '_', i)) for i in df.columns}
df.rename(columns=col_mapping, inplace=True)
df = df.loc[(df['N_stack'] <= 3200) & (df['N_sku'] == 15)]
df['T'] = df['T_w'] + df['T_s']
for i in ['U_BP', 'U_TC', 'MR_per', 'Reshuffle_time_part']:
    df[i] = df[i].apply(lambda x: 0 if x in ['-'] else x[:-1]).astype(float) / 100

# multi = df.loc[(df['N_stack'] <= 3200) & (df['N_sku'] == 15) & (df['N_BP'] == 6) & (df['T_w'] < 1000)]
# multi_nondetermined = multi.loc[(multi['storage_policy'] != 'determined')]

X_vars = ['storage_policy', 'shape', 'N_BP']
for Y in ['T', 'T_w', 'T_s', 'RECO',
          'TREC', 'SCEO', 'TSEC', 'TEC', 'U_BP', 'U_TC', 'MR_per', 'MRTiers',
          'Reshuffle_time_part', 'queue_length']:
    print(f"Y_var: {Y}")
    anova_res = anova_var(Y_var=Y, X_vars=X_vars, df=df)
    print(anova_res.to_latex(index=True))

    for x in X_vars:
        pair_res = pairwise_tukeyhsd(df[Y], df[x])
        print(pair_res)