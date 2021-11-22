# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/11/6 08:50
"""

import numpy as np
import pandas as pd

from ORCSRS.Config import w_up, w_low
from Orders.OrderEntry import OrderLine


def process_GDS_orders():
    df = pd.read_csv(filepath_or_buffer='../../Resource/GDS_orders.csv', engine='python', encoding='utf8')
    sku = df['sku_id'].value_counts()[df['sku_id'].value_counts().index.str.startswith('2.G.')]
    df = df.loc[df['sku_id'].isin(sku.index), ['订单号', 'sku_id', '数量']]
    df['sku_id'] = df['sku_id'].map(dict(zip(df['sku_id'].value_counts().index, list(range(df['sku_id'].value_counts().size)))))
    df.to_csv('../../Resource/orders.csv', index=False)


def analyze_orders(orders):
    df = pd.read_csv(filepath_or_buffer=orders, engine='python', encoding='utf8')
    pass


def bucket(sku_qty=50, weight_low=w_low, weight_up=w_up) -> pd.CategoricalDtype:
    df = pd.read_csv(filepath_or_buffer='../../Resource/orders.csv', engine='python', encoding='utf8')
    bins = np.arange(0, df['sku'].max(), df['sku'].max() // sku_qty)
    bins[-1] = df['sku'].max()
    df_bin = pd.cut(df['sku'], bins=bins, right=True, labels=np.arange(sku_qty), include_lowest=True, ordered=True)
    return df_bin


def random_orders(df_bin):
    from numpy.random import default_rng
    g = default_rng(seed=42)
    while True:
        sku_id = g.choice(df_bin)
        # yield sku_id
        yield OrderLine(sku_id + 1, w_up)


if __name__ == '__main__':
    # analyze_orders('../../Resource/orders.csv')
    m = random_orders(bucket(15))
    for i in range(10):
        print(next(m))
