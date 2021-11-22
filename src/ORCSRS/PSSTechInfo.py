# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/14 15:22
"""
# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/3/9 08:49
"""
import math
from ORCSRS.Config import *

time_load_or_unload = 1.2
time_pick_up = [5, 15]

gravity_coefficient = 9.8  # m/s^2
# c_r
friction_coefficient = 0.1
# f_r
inertial_force_coefficient = 1.15
# efficiency of
eta = 0.8

Bin_weight = 30  # kg


class PSBDemo:
    """
    PSB parameters of velocity, weight, ... in WhaleHouse.

    Transport procedure:
        1. Receive PMS command, go to the Retrieval target_xyz.
        2. Down to the designated height, seize the bin.
        3. Up to primal target_xyz.
        4. get bin to the workstation.
        5. Down to the designated height and back up.

    """
    weight = 72  # kg
    max_load_Weight = 30  # kg
    weight_of_belt = 5  # kg
    acc_h_loaded = 1.8
    acc_h_unloaded = 2
    acc_v_loaded = 1.8
    acc_v_unloaded = 2
    max_h_velocity = 2
    max_v_velocity = 1
    vertical_speed = 0.5  # m/s
    h_dis_sign = max_h_velocity * max_h_velocity / acc_h_loaded
    v_dis_sign = max_v_velocity * max_v_velocity / acc_v_loaded
    max_stretch_length = 2000 / 1000  # mm->m
    width = 645 / 1000
    length = 480 / 1000
    height = 295 / 1000
    average_power = 120  # W/h

    # calculate data
    pick_efficiency = 20  # (bins/hour/per psb), 20% fluctuation
    inbound_efficiency = 40  # (bins/hour/per psb), 20% fluctuation

    # catch up and drop off setup time
    setup_of_catchup = 14
    setup_of_dropoff = 17
    vertical_move_coefficient = 1.0


class PSTDemo:
    """
    PST parameters if velocity, weight, available load weight, acceleration... in WhaleHouse.

    Change Line procedure:
        1. Receive PMS command, go to designated ChangeLine point to wait.
        2. PSB gets to the designated ChangeLine point
        3. PST Locks the PSB and gets it to ORCSRS designated Release point.
        4. PST Releases PSB, PSB get rid of PST and enter another line.
    """
    weight = 70  # kg
    max_load_weight = 100  # kg
    max_velocity = 2  # m/s
    acc_h_loaded = 1.5  # m**2/s
    acc_h_unloaded = 1  # m**2/s
    width = 650 / 1000
    length = 750 / 1000
    height = 365 / 1000
    max_h_velocity = 2
    h_dis_sign = max_h_velocity * max_h_velocity / acc_h_loaded
    # calculate data
    efficiency_change_line = 150  # times/hour/per pst, 20% fluctuation


class BinDemo:
    WH9637_stack_height = 360 / 1000

    """
    Provide two recommended bin size as following.
    EU4628, WH9637
    """
    # bin standards
    # EU4628 outer size (mm)
    EU4628_l_outer = 600 / 1000
    EU4628_w_outer = 400 / 1000
    EU4628_h_outer = 280 / 1000
    # EU4628 inner size (mm)
    EU4628_l_inner = 565 / 1000
    EU4628_w_inner = 365 / 1000
    EU4628_h_inner = 268 / 1000
    # EU4628 stack height (mm)
    EU4628_stack_height = 270 / 1000

    # WH9637 outer size (mm)
    WH9637_l_outer = 900 / 1000
    WH9637_w_outer = 600 / 1000
    WH9637_h_outer = 370 / 1000
    # WH9637 inner size (mm)
    WH9637_l_inner = 852 / 1000
    WH9637_w_inner = 552 / 1000
    WH9637_h_inner = 340 / 1000
    # WH9637 stack height (mm)


class StackDemo:
    """
    we set a twin data saver for the lag of picking.
    """
    height_of_one_tier = 0.33
    tolerate_coefficient = 1  # 当前系数可能是80%
    length = 0.6  # 俯视图上的0。6m每个推塔
    width = 1  # 记录为每两行货轨直接的中心距


class Warehouse:
    """
    Define a Warehouse class to get an available target_xyz and to save parameters of warehouse configuration.
    """
    # num_of_cols = 10
    # stacks_of_one_col = 50
    height_of_one_tier = .33
    # num_of_tiers = 8
    init_rate = 0.5


class WorkStation:
    pass
