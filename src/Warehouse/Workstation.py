# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/6/14 14:18
"""
from simpy import Resource, Environment


class WorkStation(Resource):

    def __init__(self, env: Environment):
        super().__init__(env)

    def pickup(self):
        return 1.2

