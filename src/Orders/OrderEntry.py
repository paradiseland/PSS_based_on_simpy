# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/28 23:58
"""
import logging
from collections import OrderedDict
from typing import Optional, Tuple, Union, Dict, List, TYPE_CHECKING

import numpy as np
from simpy.core import SimTime

if TYPE_CHECKING:
    from Robots.PSB import PSB
    from Robots.PST import PST


class OrderEntry:
    """
    """

    def __init__(self, target_xyz: Tuple[int, int, int] = (), s: np.ndarray = None, target=None, start=None, sku_id=None, **kwargs):
        self.name = ""
        self.arrive_time: SimTime = 0
        self.start_time: SimTime = 0
        self.end_time: SimTime = 0
        self.sku_id: int = sku_id
        self.robots: Dict[str, Optional[Union[PSB, PST]]] = {'PSB': None, 'PST': None}
        self.start_stack: Optional[np.ndarray] = s
        self.end_stack: Optional[np.ndarray] = None
        self.o = start
        self.d = target
        self.time_line = OrderedDict()
        self.target_xyz = target_xyz
        self._reshuffle = np.sum(s > 0) - target_xyz[2] - 1 if s is not None and target_xyz else -1
        self.ec = {'PSB': 0, 'PSB_reshuffle': 0, 'PST': 0}
        self.reshuffle_time = 0
        self.__dict__.update(kwargs)

    def update(self, **kwargs):
        self.__dict__.update(kwargs)

    def bind(self, robot):  # : Union[PST, PSB]
        from Robots.PSB import PSB
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
        # except TypeError:
        #     logging.error(f"{self.name}")

    @property
    def waiting_time(self):
        return self.start_time - self.arrive_time

    def has_waiting_since(self, time):
        return int(time - self.arrive_time)

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
            logging.info(f'{self.name}, reshuffle task <= 0, {self.target_xyz}, stack info:{self.start_stack} -> {self.end_stack}')


class InboundOrderEntry(OrderEntry):
    type_ = 'S'

    def __init__(self, name, target_xyz: Tuple[int, int, int] = (), s: np.ndarray = None, **kwargs):
        super().__init__(target_xyz, s, **kwargs)
        self.d = target_xyz[:2] if target_xyz else None
        self.name = name


class OutboundOrderEntry(OrderEntry):
    type_ = 'R'

    def __init__(self, name, target_xyz: Tuple[int, int, int] = (), s: np.ndarray = None, **kwargs):
        super().__init__(target_xyz, s, **kwargs)
        # self.reshuffle = np.sum(s) - target_xyz[2] - 1
        self.d = target_xyz[:2] if target_xyz else None
        self.name = name

    def __str__(self):
        return f"{self.name},[{self.d}]"


class OrderLine:
    def __init__(self, sku_id, quantity):
        self.sku_id = sku_id
        self.qty = quantity

    def __str__(self):
        return f"sku[{self.sku_id}], quantity:[{self.qty}]"


class Order:
    """

    """

    def __init__(self, orderlines):
        self.entries: List[OrderEntry] = orderlines


class InboundOrder(Order):
    """

    """

    def __init__(self, orderlines):
        super().__init__(orderlines)


class OutboundOrder(Order):
    """

    """

    def __init__(self, orderlines):
        super().__init__(orderlines)


class BatchOrders:
    """

    """
    pass


class OrderPool:
    """

    """
    pass


class RLOrderPool:
    pass
