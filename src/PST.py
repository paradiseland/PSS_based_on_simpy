# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/3/9 08:41
"""
from simpy import Resource, Environment


class PST:
    """

    """

    def __init__(self, env: Environment):
        self.env = env
        self.idle = True
        self.resource = Resource(env, capacity=1)
        self.line = 0

    def move_to_target_line(self, target_line):
        pass

    def transport_PSB_to_target_line(self, target_line, psb):

        self.released()
        pass

    def occupied(self):
        self.idle = False

    def released(self):
        self.idle = True
