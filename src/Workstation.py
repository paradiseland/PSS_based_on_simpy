# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/3/11 14:10
"""
import random
from collections import OrderedDict

from simpy import Resource, Environment

from CONSTANT import time_pick_up


class Workstation:
    """

    """

    def __init__(self, i, env: Environment):
        self.index = i
        self.resource = Resource(env=env, capacity=1)
        self.place = (i, -1)

    @staticmethod
    def get_time_of_pick_up():
        return random.uniform(time_pick_up[0], time_pick_up[1])


class Workstations:
    """

    """

    def __init__(self, workstations):  # : OrderedDict[int, Workstation]
        self.workstations = workstations
