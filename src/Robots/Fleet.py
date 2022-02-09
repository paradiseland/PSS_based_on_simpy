# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/14 10:22
"""
import logging
from typing import List, Callable, Any, TYPE_CHECKING

from simpy import FilterStore, Environment
from simpy.resources.store import FilterStoreGet, StorePut
import numpy as np
from Robots.PSB import PSB
from Robots.PST import PST

if TYPE_CHECKING:
    from ORCSRS.PSS import PSS


class Fleet(FilterStore):
    """
    定义一个psb车队
    """

    def __init__(self, env: 'PSS', qty):
        super().__init__(env)
        self.env = env
        self.all_PSBs: List[PSB] = [PSB(i, i) for i in range(qty)]
        self.items.extend(self.all_PSBs)
        for psb in self.all_PSBs:
            psb.bind_fleet(self)
        logging.info(
            f"BP robots fleet initializing...\nIncluding {len(self.all_PSBs)} BP robots, initialized at {[f'{pst.name}:{pst.place}' for pst in self.all_PSBs]}")

    @property
    def at_lines(self):
        return {psb.name: psb.line for psb in self.all_PSBs}

    def get_line(self, request_line, filter_: Callable[[Any], bool] = lambda item: True) -> FilterStoreGet:
        self.items.sort(key=lambda psb: abs(psb.line - request_line))
        return super().get(filter_)

    def get_bp_(self, name):
        """
        according by bp robot name, give a request.
        :param name: BP name
        :return:
        """
        return super().get(lambda bp: bp.name == name)

    def get_bp(self):
        return super().get()

    def __str__(self) -> str:
        return f"{len(self.items)} PSBs currently available, line: {[psb.line for psb in self.all_PSBs]}"

    def __len__(self):
        return len(self.items)

    def put(self, item: Any) -> StorePut:
        store_put = super().put(item)
        self.env.bp_fleet_activate.succeed()
        return store_put


class Fleet_PST(FilterStore):
    def __init__(self, env: 'PSS', PSTs_info):
        super().__init__(env)
        self.all_PSTs = [PST(config) for config in PSTs_info]
        self.items.extend(self.all_PSTs)
        self.request_line = 0
        logging.info(
            f"TC Robots fleet initializing...\nIncluding {len(self.all_PSTs)} tc robots, initialized at {[f'{pst.name}:{pst.place}' for pst in self.all_PSTs]}")

    def get_line(self, request_line_od, filter_: Callable[[Any], bool] = lambda item: True) -> FilterStoreGet:
        self.items.sort(key=lambda pst: (not pst.can_handle(request_line_od), abs(pst.line - request_line_od[0])))
        return super().get(filter_)

    @property
    def at_lines(self):
        return {pst.name: (pst.line, pst.side) for pst in self.all_PSTs}

    def __str__(self) -> str:
        return f"{len(self.items)} PSBs currently available."


if __name__ == '__main__':
    pass
