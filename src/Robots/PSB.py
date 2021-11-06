# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/8 12:54
"""
import copy
import logging
from functools import lru_cache
from typing import Optional, Tuple, Union, Any

import numpy as np

from ORCSRS.PSSTechInfo import PSBDemo, inertial_force_coefficient, gravity_coefficient, friction_coefficient, eta, Bin_weight, StackDemo, time_load_or_unload
from Warehouse.Stocks import Stocks
from ORCSRS.Config import *


# F_c = G/g•a•f_r + G•c_r
# P_c = F_c • max_h_velocity / (1000 * eta)
logger = logging.getLogger(__name__)

@lru_cache(10)
def power_of_horizontal_acceleration_unloaded(v_top):
    return (PSBDemo.weight * PSBDemo.acc_h_unloaded * inertial_force_coefficient + PSBDemo.weight * gravity_coefficient * friction_coefficient) * v_top / (1000 * eta)  # 0.77904


@lru_cache(10)
def power_of_horizontal_acceleration_loaded(v_top) -> float:
    return ((PSBDemo.weight + Bin_weight) * PSBDemo.acc_h_loaded * inertial_force_coefficient + (PSBDemo.weight + Bin_weight) * gravity_coefficient * friction_coefficient) * v_top / (1000 * eta)


@lru_cache(10)
def power_of_horizontal_deceleration_unloaded(v_top) -> float:
    return (PSBDemo.weight * PSBDemo.acc_h_unloaded * inertial_force_coefficient - PSBDemo.weight * gravity_coefficient * friction_coefficient) * v_top / (1000 * eta)


@lru_cache(10)
def power_of_horizontal_deceleration_loaded(v_top) -> float:
    return ((PSBDemo.weight + Bin_weight) * PSBDemo.acc_h_loaded * inertial_force_coefficient - (PSBDemo.weight + Bin_weight) * gravity_coefficient * friction_coefficient) * v_top / (1000 * eta)


@lru_cache(10)
def power_of_horizontal_constant_unloaded(v_max) -> float:
    return (PSBDemo.weight * gravity_coefficient * friction_coefficient * v_max) / (1000 * eta)


@lru_cache(10)
def power_of_horizontal_constant_loaded(v_max) -> float:
    return ((PSBDemo.weight + Bin_weight) * gravity_coefficient * friction_coefficient * v_max) / (1000 * eta)


@lru_cache(10)
def power_of_vertical_acceleration_unloaded(v_top) -> float:
    return (PSBDemo.weight * PSBDemo.acc_v_unloaded * inertial_force_coefficient + PSBDemo.weight * gravity_coefficient * friction_coefficient) * v_top / (1000 * eta)  # 0.77904


@lru_cache(10)
def power_of_vertical_acceleration_loaded(v_top) -> float:
    return ((PSBDemo.weight + Bin_weight) * PSBDemo.acc_v_unloaded * inertial_force_coefficient + (PSBDemo.weight + Bin_weight) * gravity_coefficient * friction_coefficient) * v_top / (
                1000 * eta)  # 0.77904


@lru_cache(10)
def power_of_vertical_deceleration_unloaded(v_top) -> float:
    return (PSBDemo.weight * PSBDemo.acc_v_unloaded * inertial_force_coefficient + PSBDemo.weight * gravity_coefficient * friction_coefficient) * v_top / (1000 * eta)  # 0.77904


@lru_cache(10)
def power_of_vertical_deceleration_loaded(v_top) -> float:
    return ((PSBDemo.weight + Bin_weight) * PSBDemo.acc_v_unloaded * inertial_force_coefficient + (PSBDemo.weight + Bin_weight) * gravity_coefficient * friction_coefficient) * v_top / (
                1000 * eta)  # 0.77904


@lru_cache(10)
def power_of_vertical_constant_unloaded(v_max) -> float:
    return (PSBDemo.weight * gravity_coefficient * v_max) / (1000 * eta)


@lru_cache(10)
def power_of_vertical_constant_loaded(v_max) -> float:
    return ((PSBDemo.weight + Bin_weight) * gravity_coefficient * v_max) / (1000 * eta)


# -----------------------------------------------------------------------------


class PSB:
    """
    不继承任何东西, 作为FilterStore[车队]的元素
    """

    def __init__(self, ID: int, init_line=None):
        self.ID: int = ID
        self.name = f"PSB-{ID}"
        self._line: Optional[int] = init_line
        # 单轨的调整位置仅有出发点和目标点，舍弃翻箱的位置变化记录
        self._place: [Tuple[int, int]] = (self.line, -1)  # 初始化在默认的工作台(line, -1)位置
        self.idle: bool = True
        self.energy_consumption_for_current_order: float = 0
        self.total_work_time: float = 0.0
        self.working_time = 0
        self.reshuffle_task = []  # 这个返回的任务应该绑定工作台而不是绑定psb
        self.cur_order: Optional['Order'] = None
        self.fleet: Optional['Fleet'] = None

    @property
    def place(self):
        return self._place

    @place.setter
    def place(self, p):
        if p[1] < -1 or p[1] > STACKS_OF_ONE_COL :
            logger.error(f"{p}, PSB位置设定有误")
        elif not isinstance(p, tuple):
            logger.error(f"位置变更输入非元组")
        else:
            self._place = p
            self.line = p[0]

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, value):
        tmp = copy.deepcopy(self.fleet.at_lines)
        tmp.pop(self.name)
        if value not in tmp.values():
            self._line = value
        else:
            logger.info(f"该道已有psb, {self.fleet.at_lines}, PSB-{self.ID}->line{value}")

    def bind_fleet(self, fleet):
        self.fleet = fleet

    def go_to_horizontally(self, target: Tuple[int, int], is_loaded=False) -> Union[float, Any]:
        """
        compute the time consumption from current place to target place.
        ---------------
        :param target: current place -> target place: (x, y) coordinates of the stack place. x in width and y in length.
        :param is_loaded: whether carrying load or not
        :return 返回水平方向前往目的地的时间
        """
        accelerate_time, stable_move_time, time_cur_place2target = self.get_horizontal_transport_time_and_ec(abs(target[1] - self.place[1]) * StackDemo.length, is_loaded)
        return time_cur_place2target

    def get_horizontal_transport_time_and_ec(self, dis: int, is_loaded=False, is_reshuffle=False):
        """
        Return the acc time, stable time and total time to finish this distance.
        :param dis: 水平移动距离
        :param is_loaded: 是否负载
        """
        if dis < PSBDemo.h_dis_sign:
            t_1 = (dis / PSBDemo.acc_h_loaded) ** .5  # 1/2 dis = 1/2 at^2 -> dis = at^2
            v_top = t_1 * PSBDemo.acc_h_loaded
            work = PSB.get_horizontal_work_ACC(t_1, v_top, loaded=is_loaded)
            assert work >= 0, "能耗为负"
            t_2 = 0
        else:
            t_1 = PSBDemo.max_h_velocity / PSBDemo.acc_h_loaded
            t_2 = (dis - PSBDemo.h_dis_sign) / PSBDemo.max_h_velocity
            work = PSB.get_horizontal_work_ACC(t_1, PSBDemo.max_h_velocity, stable_time=t_2, loaded=is_loaded)
            assert work > 0, "能耗为负"
        self.update_energy_consumption_for_current_order(work, is_reshuffle)
        return t_1, t_2, 2 * t_1 + t_2

    def get_vertical_transport_time_JC(self, tier: int, drop_off=True, is_reshuffle=False):
        """
        Modified by Xingwei Chen 2021/4/6 16:31
        Based on time study in PSS system, we get the law of the catch and drop off bins.
        --------
        example: catch-up: {'Tier-0': 21, 'Tier-1': 20, 'Tier-2': 19, 'Tier-3': 18, 'Tier-4': 17, 'Tier-5': 16, 'Tier-6': 15}s
                 drop-off: {'Tier-0': 24, 'Tier-1': 23, 'Tier-2': 22, 'Tier-3': 21, 'Tier-4': 20, 'Tier-5': 19, 'Tier-6': 18}
        :param tier: target tier of this stack
        :param drop_off: whether this step is dropping of current bin
        """
        #  Modified by Xingwei Chen 2021/4/6 16:06
        #  Modified by Xingwei Chen 2021/4/11 12:36
        work = PSB.get_vertical_work_JC(tier)
        self.update_energy_consumption_for_current_order(work, is_reshuffle)
        if drop_off:
            return PSBDemo.setup_of_dropoff + (NUM_OF_TIERS - tier) * PSBDemo.vertical_move_coefficient
        else:
            return PSBDemo.setup_of_catchup + (NUM_OF_TIERS - tier) * PSBDemo.vertical_move_coefficient

    def get_vertical_transport_time_ACC(self, tier: int, drop_off=True, is_reshuffle=False) -> Tuple[float, float, float]:
        """
        默认当前为顶层
        :param tier: 目标层数
        :param drop_off: == loaded
        :return:
        """
        dis = (NUM_OF_TIERS - tier) * StackDemo.height_of_one_tier
        acc = PSBDemo.acc_v_loaded if drop_off else PSBDemo.acc_v_unloaded
        if dis <= PSBDemo.v_dis_sign:
            t_1 = (float(dis / acc)) ** .5
            t_2 = 0
        else:
            t_1 = PSBDemo.max_v_velocity / acc
            t_2 = (dis - PSBDemo.v_dis_sign) / PSBDemo.max_v_velocity
        work = PSB.get_vertical_work_ACC(tier, t_1 * acc, loaded=drop_off)
        self.update_energy_consumption_for_current_order(work, is_reshuffle)
        return t_1, t_2, 2 * t_1 + t_2

    def occupied_by(self, order):
        self.idle = False
        self.bind(order)
        self.energy_consumption_for_current_order = 0

    def released(self):
        self.idle = True

    def reshuffle_bin_to_destination(self, pre_y, pre_tier, new_y, new_tier):
        """
        transport the blocking bins to peek of adjacent stack.
        the current place of PSB doesn't change
        同一个轨道中的翻箱任务, 起点终点均为原始点, Warning:[会回到起始点]
          → → → → → → → → → → → → → → →
          ↑ ← ← ← ← ← ← ← ← ← ← ← ← ↑ ↓
          ↑ ↓                       ↑ ↓
          ↑ ↓                       ↑ ↓
          ↑ ↓                       ↑ ↓
        此处假定了
        :param pre_y 出发点的y
        :param pre_tier 出发点的层数
        :param new_y 目的点的y
        :param new_tier 目的点的层数
        """
        down_up_to_peek = self.get_vertical_transport_time_JC(pre_tier, drop_off=False, is_reshuffle=True)
        horizon_to_back_adjacent = self.get_horizontal_transport_time_and_ec((abs(pre_y - new_y)) * StackDemo.length, is_reshuffle=True)[-1] * 2  # 水平移动时间, 包括回程时间
        drop_off_up = self.get_vertical_transport_time_JC(new_tier, drop_off=True, is_reshuffle=True)
        total_time = down_up_to_peek + horizon_to_back_adjacent + drop_off_up + time_load_or_unload * 2
        self.cur_order.reshuffle_time += total_time
        return total_time

    def retrieve2workstation_without_shuffling(self, warehouse, xy, tier):
        """
        不带shuffle的取货流程, 从目标推塔向下取货然后送往工作台
        :param warehouse:
        :param tier: 堆塔的层
        :param xy: 堆塔在一列中的y位置
        :return: 总时间
        """
        x, y = xy
        down_and_up_to_peek = self.get_vertical_transport_time_JC(tier, drop_off=False)
        warehouse.stocks.R(x, y, int(warehouse.stocks.s[x, y].sum() - 1))
        stack2workstation_horizontally = self.get_horizontal_transport_time_and_ec(y * StackDemo.length)[-1]
        drop_off_up = self.get_vertical_transport_time_JC(tier, drop_off=True)
        total_time = down_and_up_to_peek + stack2workstation_horizontally + drop_off_up + time_load_or_unload * 2  # t_lu: time of loading or unloading
        return total_time

    def reshuffle_and_get_bin(self, warehouse: 'PSS', xy: tuple, tier, method: str):
        if method == 'without':
            return self.get_bin_without_return(warehouse, xy, tier)
        elif method == 'immediate':
            return self.get_bin_with_immediate_return(warehouse, xy, tier)
        elif method == 'delayed':
            return self.get_bin_with_delayed_return()
        else:
            raise TypeError("错误的翻箱策略输入")

    def get_bin_with_immediate_return(self, warehouse, xy, stack_tier):
        """
        immediate return
        reshuffle -> target bin go to temporarily place -> return blocking bins -> go to workstation.
        """
        pass

    def get_bin_without_return(self, warehouse: 'PSS', xy, tier):
        """
        不带return的一次取货流程
        :param warehouse:
        :param xy:
        :param tier: 从0开始计数
        :return:
        """
        target_bin_new_stack_y, time_reshuffle = self.reshuffle_blocking_bin(warehouse, xy, tier)
        return time_reshuffle

    def get_bin_with_delayed_return(self):
        pass

    def reshuffle_blocking_bin(self, warehouse: 'PSS', xy: tuple, tier: int):
        """
        by reshuffle the bin of certain stack, get the target bin blocking by other bins.
        """
        x, y = xy
        this_stack: np.ndarray = warehouse.stocks.s[x, y].view()
        adjacent_stacks = [n for m in [(y + i, y - i) for i in range(1, STACKS_OF_ONE_COL)] for n in m if
                           0 <= n < STACKS_OF_ONE_COL]  # and warehouse.stocks.sync_[x, n] 此处修改的原因是，作为离散事件中的一个点，同步表是否变多没关系，只要真实表可用就行
        adjacent_place_chosen = 0  # start place of blocking bins in sequence
        time_of_reshuffle_blocking_bins = 0
        next_stack: np.ndarray = warehouse.stocks.s[x, adjacent_stacks[adjacent_place_chosen]].view()
        while this_stack.sum() > tier + 1:
            cur_bin = int(this_stack.sum() - 1)
            next_bin = next_stack.sum()
            if next_bin < NUM_OF_TIERS-1:
                # self.register_reshuffle(adjacent_stacks[adjacent_placDe_chosen])
                time_of_reshuffle_blocking_bins += self.reshuffle_bin_to_destination(
                        y, cur_bin, adjacent_stacks[adjacent_place_chosen], next_bin)
                try:
                    warehouse.stocks.R(x, y, cur_bin, is_R=False)
                    warehouse.stocks.S(x, adjacent_stacks[adjacent_place_chosen], next_bin, is_S=False)
                except AssertionError:
                    print("出入库失败")
            if next_stack.sum() >= NUM_OF_TIERS - 1:
                adjacent_place_chosen += 1
                next_stack = warehouse.stocks.s[x, adjacent_stacks[adjacent_place_chosen]].view()
        return adjacent_stacks[adjacent_place_chosen], time_of_reshuffle_blocking_bins

    def store(self, target: tuple, warehouse: 'PSS') -> float:
        # down -> up -> storage place -> down -> up
        x, y = target
        down_up_to_peek = self.get_vertical_transport_time_JC(NUM_OF_TIERS, drop_off=False)
        time_wk2storage = self.go_to_horizontally((x, y), True)
        stack = warehouse.stocks.s[x, y]
        drop_off_up = self.get_vertical_transport_time_JC(stack.sum() - 1, drop_off=True)
        total_time = down_up_to_peek + time_wk2storage + drop_off_up + time_load_or_unload * 2
        return total_time

    def register_reshuffle(self, y):
        """
        reshuffle task is recorded in the form of int of y coordinate + [last y].
        :param y:
        :return:
        """
        self.reshuffle_task.append([self.line, y])

    def return_blocking_bins(self, warehouse: 'PSS', last_y: int):
        """
        when the reshuffle task is not null, we came back and complete the reshuffling
        """
        pass

    def get_work_horizontally(self, time_acc, time_constant=0, total_weight=PSBDemo.weight):
        pass

    @staticmethod
    def get_horizontal_work_ACC(acc_time, v_top, stable_time=0, loaded=False) -> Union[int, float]:
        """
        W = P_acc * t_1 + P_dec * t_1 + P_cons * t_stable

        :param v_top:
        :param acc_time: time of PSB accelerating
        :param stable_time: time of PSB with constant speed
        :param loaded: if this PSB is carrying a bin.
        :return: work of horizontal moving.
        """
        if loaded:
            work = (power_of_horizontal_acceleration_loaded(v_top) * acc_time +
                    power_of_horizontal_deceleration_loaded(v_top) * acc_time +
                    power_of_horizontal_constant_loaded(v_top) * stable_time) / 3600
        else:
            work = (power_of_horizontal_acceleration_unloaded(v_top) * acc_time +
                    power_of_horizontal_deceleration_unloaded(v_top) * acc_time +
                    power_of_horizontal_constant_unloaded(v_top) * stable_time) / 3600
        try:
            assert work >= 0
        except AssertionError:
            print(f"{acc_time}, {v_top}, {stable_time},{loaded}, {work}")
        return work

    @staticmethod
    def get_vertical_work_JC(tier) -> Union[int, float]:
        """
        W = mgh.
        while unloaded, we compute the work by W = 1/2 (m_belt_down)gh, which using calculus.
        :param tier: [0, NUM_OF_TIERS-1]
        :return: work of this work, including down and up
        """
        work = (NUM_OF_TIERS - tier) / NUM_OF_TIERS * PSBDemo.weight_of_belt * gravity_coefficient * (
                NUM_OF_TIERS - tier) * StackDemo.height_of_one_tier + \
               Bin_weight * gravity_coefficient * (NUM_OF_TIERS - tier) * StackDemo.height_of_one_tier
        try:
            assert work >= 0
        except AssertionError:
            print(f"{tier}层有误")
        return work / 3600000

    @staticmethod
    def get_vertical_work_ACC(acc_time, v_top, stable_time=0, loaded=False) -> Union[int, float]:
        """
        W = P_acc * t_1 + P_dec * t_1 + P_cons * t_stable

        :param v_top:
        :param acc_time: time of PSB accelerating
        :param stable_time: time of PSB with constant speed
        :param loaded: if this PSB is carrying a bin.
        :return: work of horizontal moving.
        """
        if loaded:
            work = (power_of_vertical_acceleration_loaded(v_top) * acc_time +
                    power_of_vertical_deceleration_loaded(v_top) * acc_time +
                    power_of_vertical_constant_loaded(v_top) * stable_time) / 3600
        else:
            work = (power_of_vertical_acceleration_unloaded(v_top) * acc_time +
                    power_of_vertical_deceleration_unloaded(v_top) * acc_time +
                    power_of_vertical_constant_unloaded(v_top) * stable_time) / 3600
        try:
            assert work >= 0
        except AssertionError:
            print(f"{acc_time}, {v_top}, {stable_time},{loaded}, {work}")
        return work

    def update_energy_consumption_for_current_order(self, new_consumption, is_reshuffle=False):
        # print("ec: " + str(new_consumption))
        self.cur_order.ec['PSB'] += new_consumption
        if is_reshuffle:
            self.cur_order.ec['PSB_reshuffle'] += new_consumption
        self.energy_consumption_for_current_order += new_consumption

    def bind(self, order):
        order.bind(self)
        self.cur_order = order

    def __str__(self) -> str:
        return f"PSB-{self.ID}:{'idle' if self.idle else 'busy'} line[{self.line}], {self.place}."



if __name__ == '__main__':
    from ORCSRS.PSS import PSS

    test = Stocks(shape=(10, 20, 7), rate=0.5)
    pss = PSS()
    psb = PSB(3, 3)
    max_stack = np.argmax(pss.stocks.s_2d[3])
    psb.reshuffle_blocking_bin(pss, (3, max_stack), 2)
    # psb1 = PSB(1, 1)
    # psb2 = PSB(2, 2)
    # print(psb1)
