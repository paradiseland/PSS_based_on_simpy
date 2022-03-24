# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2022/2/9 11:55
"""

# !/usr/bin/env python
# coding=utf-8
import pandas as pd
import scipy.stats as st

'''
Author: John
Email: johnjim0816@gmail.com
Date: 2021-03-12 16:02:24
LastEditor: John
LastEditTime: 2021-11-30 18:39:19
Discription: 
Environment: 
'''
import os
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from matplotlib.font_manager import FontProperties  # 导入字体模块


def chinese_font():
    ''' 设置中文字体，注意需要根据自己电脑情况更改字体路径，否则还是默认的字体
    '''
    try:
        font = FontProperties(
            fname='/System/Library/Fonts/STHeiti Light.ttc', size=15)  # fname系统字体路径，此处是mac的
    except:
        font = None
    return font


def plot_rewards_cn(rewards, ma_rewards, plot_cfg, tag='train'):
    ''' 中文画图
    '''
    sns.set()
    plt.figure()
    plt.title(u"{}环境下{}算法的学习曲线".format(plot_cfg.env_name,
                                       plot_cfg.algo_name), fontproperties=chinese_font())
    plt.xlabel(u'回合数', fontproperties=chinese_font())
    plt.plot(rewards)
    plt.plot(ma_rewards)
    plt.legend((u'奖励', u'滑动平均奖励',), loc="best", prop=chinese_font())
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path + f"{tag}_rewards_curve_cn")
    # plt.show()


def plot_rewards(rewards, ma_rewards, plot_cfg, tag='train'):
    sns.set()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    # plt.title("learning curve on {} of {} for {}".format(
        # plot_cfg.device, plot_cfg.algo_name, plot_cfg.env_name))
    plt.xlabel('回合')
    plt.plot(rewards, label='奖励')
    plt.plot(ma_rewards, label='每10回合平均奖励')
    plt.legend()
    if plot_cfg.save:
        plt.savefig(plot_cfg.result_path + "{}_rewards_curve".format(tag), dpi=800)
    plt.show()


def plot_losses(losses, algo="DQN", save=True, path='./'):
    sns.set()
    plt.figure()
    plt.title("loss curve of {}".format(algo))
    plt.xlabel('epsiodes')
    plt.plot(losses, label='rewards')
    plt.legend()
    if save:
        plt.savefig(path + "losses_curve")
    plt.show()


def save_results(rewards, ma_rewards, tag='train', path='./results'):
    ''' 保存奖励
    '''
    np.save(path + '{}_rewards.npy'.format(tag), rewards)
    np.save(path + '{}_ma_rewards.npy'.format(tag), ma_rewards)
    print('结果保存完毕！')


def make_dir(*paths):
    ''' 创建文件夹
    '''
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)


def del_empty_dir(*paths):
    ''' 删除目录下所有空文件夹
    '''
    for path in paths:
        dirs = os.listdir(path)
        for dir in dirs:
            if not os.listdir(os.path.join(path, dir)):
                os.removedirs(os.path.join(path, dir))


def get_ci_interval(array: pd.Series, ci=1):
    mu = array.mean(axis=0)
    std = array.std(axis=0)
    tscore = st.t.ppf(1 - ci, array.shape[0] - 1)
    lb = mu - (tscore * std / array.shape[0] ** 0.5)
    ub = mu + (tscore * std / array.shape[0] ** 0.5)
    return lb, ub
