# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/14 14:17
"""
from functools import lru_cache
from typing import Union

from ORCSRS.Config import STACKS_OF_ONE_COL
from ORCSRS.PSSTechInfo import PSTDemo, PSBDemo, StackDemo, inertial_force_coefficient, gravity_coefficient, friction_coefficient, eta, Warehouse


@lru_cache(10)
def power_of_horizontal_acceleration_unloaded(v_top):
    return (PSTDemo.weight * PSTDemo.acc_h_unloaded * inertial_force_coefficient + PSTDemo.weight * gravity_coefficient * friction_coefficient) * v_top / (1000 * eta)  # 0.77904


@lru_cache(10)
def power_of_horizontal_constant_unloaded(v_max) -> float:
    return (PSTDemo.weight * gravity_coefficient * friction_coefficient * v_max) / (1000 * eta)


@lru_cache(10)
def power_of_horizontal_acceleration_loaded(v_top) -> float:
    return ((PSTDemo.weight + PSBDemo.weight) * PSTDemo.acc_h_loaded * inertial_force_coefficient + (PSTDemo.weight + PSBDemo.weight) * gravity_coefficient * friction_coefficient) * v_top / (
            1000 * eta)


@lru_cache(10)
def power_of_horizontal_constant_loaded(v_max) -> float:
    return ((PSTDemo.weight + PSBDemo.weight) * gravity_coefficient * friction_coefficient * v_max) / (1000 * eta)


@lru_cache(10)
def power_of_horizontal_deceleration_unloaded(v_top) -> float:
    return (PSTDemo.weight * PSTDemo.acc_h_unloaded * inertial_force_coefficient - PSTDemo.weight * gravity_coefficient * friction_coefficient) * v_top / (1000 * eta)


@lru_cache(10)
def power_of_horizontal_deceleration_loaded(v_top) -> float:
    return ((PSTDemo.weight + PSBDemo.weight) * PSTDemo.acc_h_loaded * inertial_force_coefficient - (PSTDemo.weight + PSBDemo.weight) * gravity_coefficient * friction_coefficient) * v_top / (
            1000 * eta)


class PST:
    """
    two sides of pss system has psts to transferring PSB to different line.
    they dwell at different sides of pss
    example: ↓
    - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - -
     ⎸◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ️◾️ ◾️ ◾️  ⎸
    🚁◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ️◾️ ◾️ ◾️️  ⎸
     ⎸◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ️◾️ ◾️ ◾️ 🚁
    ⎻⎻◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ️◾️ ◾️ ◾️  ⎸
     ⎸ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ️◾️ ◾️   ⎸
    🚁◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ◾️ ️◾️ ◾️ ◾️  ⎸
    - - - - - - - - - - - - - - - - - - - -- - - - - - - - - - - - - - -
    """

    def __init__(self, config: dict):
        self.idle: bool = True
        self.side: bool = config['side']
        self.name = f"PST-{config['ID']}"
        self.place = (config['init_line'], -1)  # 根据side来获取place真实位置
        self.interval: list = config['interval']
        self.energy_consumption_for_current_order = 0
        self.cur_order = None
        self.working_time = 0

    @property
    def interval_length(self):
        return abs(self.interval[1] - self.interval[0])

    @property
    def place(self):
        return self._place

    @place.setter
    def place(self, xy: tuple):
        if self.side:
            self._place = xy
        else:
            self._place = (xy[0], STACKS_OF_ONE_COL)
        self.line = xy[0]

    @property
    def line(self):
        return self._line

    @line.setter
    def line(self, value):
        self._line = value

    def can_handle(self, demand):
        return all([i in self.interval for i in demand])

    def occupied_by(self, order):
        self.idle = False
        self.bind(order)
        self.energy_consumption_for_current_order = 0

    def released(self):
        self.idle = True

    def take_psb2line(self, x):
        return self.go_to_horizontally(x, is_loaded=True)

    def bind(self, order: 'OrderEntry'):
        order.bind(self)
        self.cur_order = order

    def go_to_horizontally(self, x, is_loaded=False):
        t_1, t_2, t = self.get_horizontal_transport_time_and_ec(abs(self.line - x) * StackDemo.width, is_loaded)
        return t

    def get_horizontal_transport_time_and_ec(self, dis: int, is_loaded=False):
        """
        Return the acc time, stable time and total time to finish this distance.
        :param dis: 水平移动距离
        :param is_loaded: 是否负载
        """
        if dis < PSTDemo.h_dis_sign:
            t_1 = (dis / PSTDemo.acc_h_loaded) ** .5  # 1/2 dis = 1/2 at^2 -> dis = at^2
            v_top = t_1 * PSTDemo.acc_h_loaded
            work = PST.get_horizontal_work_ACC(t_1, v_top, loaded=is_loaded)
            assert work >= 0, "能耗为负"
            t_2 = 0
        else:
            t_1 = PSTDemo.max_h_velocity / PSTDemo.acc_h_loaded
            t_2 = (dis - PSTDemo.h_dis_sign) / PSTDemo.max_h_velocity
            work = PST.get_horizontal_work_ACC(t_1, PSBDemo.max_h_velocity, stable_time=t_2, loaded=is_loaded)
            assert work > 0, "能耗为负"
        self.update_energy_consumption_for_current_order(work)
        return t_1, t_2, 2 * t_1 + t_2

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

    def update_energy_consumption_for_current_order(self, new_consumption):
        self.cur_order.ec['PST'] += new_consumption
        self.energy_consumption_for_current_order += new_consumption

    def __str__(self) -> str:
        return f"{self.name}:{'idle' if self.idle else 'busy'} line[{self.line}], {self.place}."
