# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2022/1/11 09:03
"""


class StockException(Exception):
    pass


class StockOverFlowException(StockException):
    pass


class StockSyncException(StockException):
    pass


class StockEmptyException(Exception):
    pass


class ConfigException(Exception):
    pass


class StrategyConfException(ConfigException):
    pass


class DedicatedStorageAreaFullException(Exception):
    pass
