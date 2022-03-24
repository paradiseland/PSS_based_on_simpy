# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2022/3/11 15:52
"""
import pandas as pd
from scipy import stats

from ORCSRS.Config import experiments_replication_csv

per_cols = ['U_{BP}', 'U_{TC}', 'MR_{per}', 'Reshuffle time part']
df = pd.read_csv(experiments_replication_csv, engine='python', index_col=False)
# df = df.loc[df['N_{BP}'] == 6]
df = df.loc[df['shape'] == '10×40×8']
for i in per_cols:
    df[i] = df[i].apply(lambda x: 0 if x in ['-'] else x[:-1]).astype(float) / 100
CI_cols = ['T_w(s)', 'T_s(s)',
           'RECO(kJ)', 'TREC(kJ)', 'SCEO(kJ)', 'TSEC(kJ)', 'TEC(kJ)',
           'U_{BP}', 'U_{TC}', 'MR_{per}', 'MRTiers',
           'Reshuffle time part', 'queue length']
scenario_cols = ['sku strategy', 'storage policy', 'shape', 'N_{stack}', 'N_{BP}']
doe_cols = []
df_groupby = df.groupby(['sku strategy', 'storage policy', 'N_{BP}'])
res = pd.DataFrame(columns=scenario_cols + CI_cols)
scenario = pd.DataFrame(columns=scenario_cols)
for info, group, in df_groupby:
    group_series = pd.Series(index=scenario_cols + CI_cols)
    group_series[scenario_cols] = list(info) + [group['N_{stack}'].iloc[0], group['N_{BP}'].iloc[0]]
    group_CI = group[CI_cols]
    mean = group_CI.mean()
    std_error = group_CI.std()
    for j in CI_cols:
        if j not in per_cols:
            group_series[j] = f"{mean[j]:.1f} pm {1.96 * std_error[j]:.5f}"
        else:
            group_series[j] = f"{mean[j]*100:.1f}% pm {1.96 * std_error[j]*100:.3f}%"
    res = res.append(group_series, ignore_index=True)
    res.sort_values('N_{BP}', inplace=True)
print(res.to_latex(index=False))
