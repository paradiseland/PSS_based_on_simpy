# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/8 13:38
"""
import numpy as np
import logging

# constants
RANDOM = 'random'
ZONED = 'zoned'
zoned_coe = 1.5
zoned_first_class_cumsum_percent = 0.7
MIXED = 'mixed'
TOP_FIRST = 'top_first'  # 订单命中的策略
ONE_SKU, MULTI_SKU = 'single-sku', 'multi-sku'
DETERMINED = 'determined'
EPSILON = 0.1

# log config
without_change_track_log_file_path = '../Result/1l1psb3200.log'
change_track_log_file_path = '../Result/mull1psb3200.log'
result_csv = '../Result/result.csv'

# Simulation
ARRIVAL_RATE = 100 / 3600  # Modified by Xingwei Chen 2021/7/25 16:14
EXPOVARIATE_ARRIVE = 'expovariate_arrive'
SIM_ELAPSE = 1000000
SIM_ELAPSE_RL = 100000
# Warehouse
NUM_OF_COLS = 10
STACKS_OF_ONE_COL = 40
NUM_OF_TIERS = 8
WAREHOUSE_INIT_RATE = 0.5
AVAIL_PLACE = NUM_OF_COLS * STACKS_OF_ONE_COL * NUM_OF_TIERS

# Robots
NUM_OF_PSBs = 5
NUM_OF_PSTs_left = 2
NUM_OF_PSTs_right = 1
NUM_OF_PSTs = NUM_OF_PSTs_left + NUM_OF_PSTs_right
PSTs_left = [{'ID': i, 'init_line': interval[0], 'interval': interval, 'side': True} for i, interval in
             zip(range(NUM_OF_PSTs_left), np.array_split(np.arange(NUM_OF_COLS), NUM_OF_PSTs_left))]
PSTs_right = [{'ID': NUM_OF_PSTs_left, 'init_line': interval[0], 'interval': interval, 'side': False} for i, interval in
              zip(range(NUM_OF_PSTs_right), np.array_split(np.arange(NUM_OF_COLS), NUM_OF_PSTs_right))]

# Orders
NUM_OF_SKU = 15
stock_threshold = 10
# weight
w_up = 15
w_low = 0

# Write
write_in = False  # write in Result.csv
log_level = 10  # print log content to console
# Storage policy
# store_policy = 'determined'
# store_policy = 'zoned'
store_policy = 'random'

result_cols = [
    'sim time', 'N_{sku}', 'sku strategy', 'storage policy', 'lambda', 'shape', 'N_{stack}', 'N_{BP}', 'N_{TC}',
    'R_{lock}', 'T_w(s)', 'T_s(s)', 'RECO(kJ)', 'TREC(kJ)', 'SCEO(kJ)', 'TSEC(kJ)', 'TEC(kJ)', 'U_{BP}', 'U_{TC}',
    'MR_{per}', 'MRTiers', 'Finish rate(R)', 'Finish rate(S)', 'stocks rate', 'lock rate', 'BP utility', 'TC utility',
    'R jobs', 'S jobs', 'Reshuffle time part', 'queue length'
]

# RLTask
# order pool refresh time
RL_embedded = True
order_pool_time_interval = 3600  # s
order_pool_size = 100
NUM_OF_SCHEDULING_STRATEGIES = 5
# STATE_DIM = int(4 + order_pool_size * 3 / NUM_OF_COLS + NUM_OF_TIERS * STACKS_OF_ONE_COL)
STATE_DIM = AVAIL_PLACE + order_pool_size * 3 + NUM_OF_COLS * 4
ACTION_DIM = 6

# reward
reshuffle_tiers_weight = -10
travel_length_weight = -1
change_track_weight = -5
