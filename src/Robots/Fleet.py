# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/14 10:22
"""
import logging
from typing import List, Callable, Any

from simpy import FilterStore, Environment
from simpy.resources.store import FilterStoreGet

from Robots.PSB import PSB
from Robots.PST import PST


class Fleet(FilterStore):
    """
    定义一个psb车队
    """

    def __init__(self, env: Environment, qty):
        super().__init__(env)
        self.all_PSBs: List[PSB] = [PSB(i, i) for i in range(qty)]
        self.items.extend(self.all_PSBs)
        for psb in self.all_PSBs:
            psb.bind_fleet(self)
        logging.info(f"PSB fleet initializing...\nIncluding {len(self.all_PSBs)} psts, initialized at {[f'{pst.name}:{pst.place}' for pst in self.all_PSBs]}")

    @property
    def at_lines(self):
        return {psb.name: psb.line for psb in self.all_PSBs}

    def get_line(self, request_line, filter_: Callable[[Any], bool] = lambda item: True) -> FilterStoreGet:
        self.items.sort(key=lambda psb: abs(psb.line - request_line))
        return super().get(filter_)

    def __str__(self) -> str:
        return f"{len(self.items)} PSBs currently available, line: {[psb.line for psb in self.all_PSBs]}"


class Fleet_PST(FilterStore):
    def __init__(self, env: Environment, PSTs_info):
        super().__init__(env)
        self.all_PSTs = [PST(config) for config in PSTs_info]
        self.items.extend(self.all_PSTs)
        self.request_line = 0
        logging.info(f"PST fleet initializing...\nIncluding {len(self.all_PSTs)} psts, initialized at {[f'{pst.name}:{pst.place}' for pst in self.all_PSTs]}")

    def get_line(self, request_line_od, filter_: Callable[[Any], bool] = lambda item: True) -> FilterStoreGet:
        self.items.sort(key=lambda pst: (not pst.can_handle(request_line_od), abs(pst.line - request_line_od[0])))
        return super().get(filter_)

    def __str__(self) -> str:
        return f"{len(self.items)} PSBs currently available."


def inorder_psb(order_name, env, target_line, fleet: Fleet):
    fleet.request_line = target_line
    avail_psb_store = fleet.get(lambda psb: psb.idle)
    logging.info(f"{env.now}, Order-{order_name} requests PSB")
    yield avail_psb_store
    logging.info(f"{env.now}, Order-{order_name} get this only PSB")
    psb = avail_psb_store.value
    # logging.info(f"{order_name}: {'>'.join([str(p) for p in fleet.items])}")
    yield env.timeout(10)
    logging.info(f"{env.now}, PSB-{psb.ID} is caught")
    fleet.put(psb)


def test():
    env = Environment()
    fleet = Fleet(env, [PSB(i, i) for i in range(1)])
    order = [env.process(inorder_psb(i, env, 4, fleet)) for i in range(3)]
    env.run(500)


if __name__ == '__main__':
    test()
