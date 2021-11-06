# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/8 13:38
"""
import numpy as np
import logging

# log config
log_level = logging.INFO
without_change_track_log_file_path = './1l1psb3200.log'
change_track_log_file_path = '../Result/mull1psb3200.log'

# Simulation
ARRIVAL_RATE = 120 / 3600  # Modified by Xingwei Chen 2021/7/25 16:14
SIM_ELAPSE = 1000000
# Warehouse
NUM_OF_COLS = 10
STACKS_OF_ONE_COL = 106
NUM_OF_TIERS = 3
WAREHOUSE_INIT_RATE = 0.5
AVAIL_PLACE = NUM_OF_COLS * STACKS_OF_ONE_COL * NUM_OF_TIERS

# Robots
NUM_OF_PSBs = 6
NUM_OF_PSTs_left = 2
NUM_OF_PSTs_right = 1
NUM_OF_PSTs = NUM_OF_PSTs_left + NUM_OF_PSTs_right
PSTs_left = [{'ID': i, 'init_line': interval[0], 'interval': interval, 'side': True} for i,  interval in
             zip(range(NUM_OF_PSTs_left), np.array_split(np.arange(NUM_OF_COLS), NUM_OF_PSTs_left))]
PSTs_right = [{'ID': NUM_OF_PSTs_left, 'init_line': interval[0], 'interval': interval, 'side': False} for i, interval in
              zip(range(NUM_OF_PSTs_right), np.array_split(np.arange(NUM_OF_COLS), NUM_OF_PSTs_right))]


