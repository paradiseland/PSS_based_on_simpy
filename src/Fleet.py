# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/3/9 08:42
"""
from typing import List, Dict

from src.PSB import PSB

from collections import OrderedDict

NUM_OF_PSBS = 10


class Fleet:
    """

    """

    def __init__(self, fleet: Dict[int, PSB]):
        self.psbs_record = fleet

    def has_an_idle_PSB_in_line(self, line_number: int):
        return [psb.line == line_number for psb in list(self.psbs_record.values())]
        # return self.psbs_record[line_number].idle

    def has_an_PSB_in_line(self, line_number: int) -> PSB:
        return self.psbs_record[line_number]

    def get_an_available_psb_nearly(self, line_number: int) -> PSB:
        """
        首先找到可用的，若无可用的，则选择任务数最少的一个车
        :param line_number:
        :return:
        """
        tasks = [len(psb.resource.put_queue) if psb else 1e10 for line, psb in self.psbs_record.values()]
        return self.psbs_record[tasks.index(min(tasks))]
