# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/3/9 08:40
"""
import logging
from typing import Optional, Tuple, Union

import simpy
from simpy import Resource

from CONSTANT import time_load_or_unload
from Stack import LENGTH, HEIGHT_OF_ONE_TIER, Stack
from Warehouse import NUM_OF_TIERS, HEIGHT_AVAILABLE, STACKS_OF_ONE_COL, Warehouse

HORIZONTAL_VELOCITY = 2
VERTICAL_VELOCITY = 1
HORIZONTAL_ACCELERATION = 2
VERTICAL_ACCELERATION = 1
PSB_WEIGHT = 72
WEIGHT_CAPACITY = 30
HORIZONTAL_DISTANCE_SIGN = HORIZONTAL_VELOCITY ** 2 / HORIZONTAL_ACCELERATION
VERTICAL_DISTANCE_SIGN = VERTICAL_VELOCITY ** 2 / VERTICAL_ACCELERATION


class PSB:
    """
    
    """

    def __init__(self, name, env, init_line=None, capacity=1):
        self.reshuffle_task = []
        self.name: int = name
        self.env: simpy.Environment = env
        self.resource: simpy.Resource = Resource(env, capacity)
        self.line: Optional[int] = init_line
        self.current_place: Optional[Tuple[int, int]] = (self.line, -1)
        self.idle: bool = True

    def go_to_horizontally(self, target: Tuple[int, int]) -> Union[int, float]:
        """
        compute the time consumption from current place to target place.
        ---------------
        Parameters:
        target: (x, y) coordinates of the stack place. x in width and y in length.
        """
        accelerate_time, stable_move_time, time_current_place2target = self.get_horizontal_transport_time(
            abs(target[1] - self.current_place[1]) * LENGTH)
        # TODO:  计算做工

        # self.update_dwell_point(target)
        return time_current_place2target

    @staticmethod
    def get_horizontal_transport_time(dis: int):
        """
        Return the acc time, stable time and total time to finish this distance.
        """
        if dis <= HORIZONTAL_DISTANCE_SIGN:
            t_1 = pow((dis / HORIZONTAL_ACCELERATION), 1 / 2)
            return t_1, 0, 2 * t_1
        else:
            t_1 = HORIZONTAL_VELOCITY / HORIZONTAL_ACCELERATION
            t_2 = (dis - HORIZONTAL_DISTANCE_SIGN) / HORIZONTAL_VELOCITY
            return t_1, t_2, 2 * t_1 + t_2

    @staticmethod
    def get_vertical_transport_time(dis: int):
        """
        Return the acc time, stable time and total time to finish this distance.
        """
        if dis <= VERTICAL_DISTANCE_SIGN:
            t_1 = (float(dis / VERTICAL_ACCELERATION)) ** .5
            return t_1, 0, 2 * t_1
        else:
            t_1 = VERTICAL_VELOCITY / VERTICAL_ACCELERATION
            t_2 = (dis - VERTICAL_DISTANCE_SIGN) / VERTICAL_VELOCITY
            return t_1, t_2, 2 * t_1 + t_2

    def update_dwell_point(self, target):
        self.current_place = target

    def occupied(self):
        self.idle = False

    def released(self):
        self.idle = True

    def transport_bin_to_destination(self, previous_y, new_y, previous_stack_tier, new_stack_tier):
        """
        transport the blocking bins to peek of adjacent stack.
        the current place of PSB doesn't change
        """
        down_up_to_peek = self.get_vertical_transport_time((NUM_OF_TIERS - previous_stack_tier) * NUM_OF_TIERS)[-1] * 2
        horizon_to_back_adjacent = self.get_horizontal_transport_time((abs(previous_y - new_y)) * LENGTH)[-1] * 2
        drop_off_up = self.get_vertical_transport_time((NUM_OF_TIERS - new_stack_tier) * HEIGHT_OF_ONE_TIER)[-1] * 2
        total_time = down_up_to_peek + horizon_to_back_adjacent + drop_off_up + time_load_or_unload * 2
        return total_time

    def get_time_retrieve_bin_without_shuffle(self, stack_tier, target_stack_order):
        down_and_up_to_peek = self.get_vertical_transport_time((NUM_OF_TIERS - stack_tier - 1) * HEIGHT_OF_ONE_TIER)[
                                  -1] * 2
        stack2workstation_horizontally = self.get_horizontal_transport_time(target_stack_order * LENGTH)[-1]
        drop_off_up = self.get_vertical_transport_time(NUM_OF_TIERS * HEIGHT_OF_ONE_TIER)[-1] * 2
        total_time = down_and_up_to_peek + stack2workstation_horizontally + drop_off_up + time_load_or_unload * 2  # t_lu: time of loading or unloading
        return total_time

    def reshuffle_and_get_bin(self, warehouse: Warehouse, xy: tuple, stack_tier, method: str):
        if method == 'without':
            return self.get_bin_without_return(warehouse, xy, stack_tier)
        elif method == 'immediate':
            return self.get_bin_with_immediate_return(warehouse, xy, stack_tier)
        elif method == 'delayed':
            return self.get_bin_with_delayed_return()
        else:
            raise TypeError

    def get_bin_with_immediate_return(self, warehouse, xy, stack_tier):
        """
        immediate return
        reshuffle -> target bin go to temporarily place -> return blocking bins -> go to workstation.
        """
        target_bin_new_stack_y, time_reshuffle = self.reshuffle_blocking_bin(warehouse, xy, stack_tier)
        # logging.debug(f"current line = {self.current_line}, target y = {target_bin_new_stack_y}, xy = {xy}")
        time_target_bin_temporarily = self.transport_bin_to_destination(
                stack_tier,
                warehouse.stock_record[self.line][target_bin_new_stack_y].size, xy[1], target_bin_new_stack_y)
        warehouse.stock_record[xy[0]][xy[1]].pop()
        warehouse.stock_record[xy[0]][target_bin_new_stack_y].push(1)
        time_return_blocking_bins = self.return_blocking_bins(warehouse, xy[1])
        time_to_temporarily = self.get_horizontal_transport_time(abs(target_bin_new_stack_y - xy[1]) * LENGTH)[
                                  -1] + time_load_or_unload * 2
        time_to_workstation = self.get_time_retrieve_bin_without_shuffle(
                warehouse.stock_record[self.line][target_bin_new_stack_y].size, target_bin_new_stack_y)
        warehouse.stock_record[xy[0]][target_bin_new_stack_y].pop()

        total_time = time_reshuffle + time_target_bin_temporarily + time_return_blocking_bins \
                     + time_to_temporarily + time_to_workstation

        return total_time

    def get_bin_without_return(self, warehouse, xy, stack_tier):
        target_bin_new_stack_y, time_reshuffle = self.reshuffle_blocking_bin(warehouse, xy, stack_tier)
        warehouse.stock_record[xy[0]][xy[1]].pop()
        time_to_workstation = self.get_time_retrieve_bin_without_shuffle(
                warehouse.stock_record[self.line][target_bin_new_stack_y].size(), target_bin_new_stack_y)
        total_time = time_reshuffle + time_to_workstation
        return total_time

    def get_bin_with_delayed_return(self):
        # TODO:
        pass

    def reshuffle_blocking_bin(self, warehouse: Warehouse, xy: tuple, stack_tier: int):
        """
        by reshuffle the bin of certain stack, get the target bin blocking by other bins.
        """
        this_stack: Stack = warehouse.stock_record[xy[0]][xy[1]]
        cur_y = xy[1]
        logging.debug("current stack size :{}".format(this_stack.size))
        logging.debug("target tier : {}".format(stack_tier))
        adjacent_stacks = [n for m in [(cur_y + i, cur_y - i) for i in range(1, STACKS_OF_ONE_COL)] for n in m if
                           0 <= n <= STACKS_OF_ONE_COL - 1]
        adjacent_place_chosen = 0  # start place of blocking bins in sequence
        time_of_reshuffle_blocking_bins = 0
        while this_stack.size != stack_tier + 1:
            next_stack = warehouse.stock_record[xy[0]][adjacent_stacks[adjacent_place_chosen]]

            current_bin = this_stack.size
            next_bin = next_stack.size + 1
            if next_stack.size < HEIGHT_AVAILABLE:
                self.register_reshuffle(adjacent_stacks[adjacent_place_chosen])
                this_stack.pop()
                next_stack.push(1)
                time_of_reshuffle_blocking_bins += self.transport_bin_to_destination(
                        cur_y, adjacent_stacks[adjacent_place_chosen], current_bin, next_bin)
            adjacent_place_chosen += 1

        return adjacent_stacks[adjacent_place_chosen], time_of_reshuffle_blocking_bins

    def store(self, target: tuple, warehouse: Warehouse):
        # down -> up -> storage place -> down -> up
        line, target_y = target
        down_up_to_peek = self.get_vertical_transport_time(NUM_OF_TIERS * HEIGHT_OF_ONE_TIER)[-1] * 2
        time_wk2storage = self.go_to_horizontally((line, target_y - 1))
        st = warehouse.stock_record[target[0]][target[1]]
        drop_off_up = self.get_vertical_transport_time((NUM_OF_TIERS - (st.size + 1)) * HEIGHT_OF_ONE_TIER)[-1] * 2
        total_time = down_up_to_peek + time_wk2storage + drop_off_up + time_load_or_unload * 2
        return total_time

    def register_reshuffle(self, y):
        """
        reshuffle task is recorded in the form of int of y coordinate + [last y].
        :param y:
        :return:
        """
        self.reshuffle_task.append([self.line, y])

    def return_blocking_bins(self, warehouse: Warehouse, last_y: int):
        """
                when the reshuffle task is not null, we came back and complete the reshuffling
                """
        # previous_y = self.reshuffle_task.pop()
        previous_stack = warehouse.stock_record[self.line][last_y]
        time_return_blocking = 0

        while len(self.reshuffle_task) > 0:
            line, next_y = self.reshuffle_task.pop()
            warehouse.stock_record[self.line][next_y].pop()
            previous_stack.push(1)
            new_stack_tier = previous_stack.size
            time_return_blocking += self.transport_bin_to_destination(
                    next_y, last_y, warehouse.stock_record[self.line][next_y].size + 1, new_stack_tier)
        return time_return_blocking

