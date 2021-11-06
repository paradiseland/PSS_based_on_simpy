# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/28 23:58
"""
import logging
from collections import OrderedDict
from typing import Optional, Tuple, Union, Dict

import numpy as np
from simpy.core import SimTime

from Robots.PSB import PSB
from Robots.PST import PST


class Order:
    """

    """

    def __init__(self, place: Tuple[int, int, int], s: np.ndarray, target=None, start=None):
        self.name = ""
        self.arrive_time: SimTime = 0
        self.start_time: SimTime = 0
        self.end_time: SimTime = 0
        self.sku: int = -1
        self.robots: Dict[str, Optional[Union[PSB, PST]]] = {'PSB': None, 'PST': None}
        self.start_stack: np.ndarray = s
        self.o = start
        self.d = target
        self.end_stack: Optional[np.ndarray] = None
        self.time_line = OrderedDict()
        self.place = place
        self._reshuffle = np.sum(s) - place[2] - 1
        self.ec = {'PSB': 0, 'PSB_reshuffle': 0, 'PST': 0}
        self.change_track = 0
        self.reshuffle_time = 0

    def bind(self, robot: Union[PST, PSB]):
        if isinstance(robot, PSB):
            self.robots['PSB'] = robot
        else:
            self.robots['PST'] = robot

    def arrive_at(self, time: SimTime):
        self.arrive_time = time

    def start_at(self, time: SimTime, origin):
        self.start_time = time
        self.o = origin

    def end_at(self, time: SimTime):
        self.end_time = time
        self.robots['PSB'].released()
        pst = f"{self.robots['PST'].name}" if self.robots['PST'] is not None else ''
        logging.info(
                f"{self.name:<8}, arr: {self.arrive_time:<10.2f}, start: {self.start_time:<10.2f}, end: {self.end_time:<10.2f}. waiting time: {self.waiting_time:<6.2f}s, working time: {self.executing_time:<6.2f}s --------------PSB-{self.robots['PSB'].ID:02} ec:{self.all_ec * 3600:<6.2f} KJ, [{self.o[0]},{self.o[1]}]→[{self.d[0]},{self.d[1]}], {pst}")

    @property
    def waiting_time(self):
        return self.start_time - self.arrive_time

    @property
    def executing_time(self):
        return self.end_time - self.start_time

    @property
    def all_ec(self) -> float:
        return sum([self.ec['PSB'], self.ec['PST']])

    @property
    def reshuffle(self):
        return self._reshuffle

    @reshuffle.setter
    def reshuffle(self, _value):
        if _value >= 1:
            self._reshuffle = _value
        else:
            logging.info(f'{self.name}, reshuffle task <= 0, {self.place}, stack info:{self.start_stack} -> {self.end_stack}')


class InboundOrder(Order):
    type_ = 'S'

    def __init__(self, name, place: Tuple[int, int, int], s: np.ndarray):
        super().__init__(place, s)
        self.d = place[:2]
        self.name = name


class OutboundOrder(Order):
    type_ = 'R'

    def __init__(self, name, place: Tuple[int, int, int], s: np.ndarray):
        super().__init__(place, s)
        self.reshuffle = np.sum(s) - place[2] - 1
        self.d = place[:2]
        self.name = name

    def __str__(self):
        return f"{self.name},[{self.d}]"
