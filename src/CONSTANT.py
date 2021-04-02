# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/3/9 08:49
"""
import math

max_stacks_num = 7
tolerate_coefficient = .7
height_available = math.ceil(max_stacks_num * tolerate_coefficient)

lines = 10
N_WORKSTATION =lines


ARRIVAL_RATE = 200/3600

time_load_or_unload = 1.2
time_pick_up = [5, 15]
