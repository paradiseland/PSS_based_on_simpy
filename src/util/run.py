# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/12/5 15:04
"""
import functools
import time
import logging


def timeit(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        local_time = time.time()
        res = func(*args, **kwargs)
        logging.log(30, f"{func.__name__}, {time.time() - local_time:.5f}s")
        return res

    return wrapper

