# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/10/19 19:42
"""
import matplotlib.pyplot as plt
import numpy as np

from Config import without_change_track_log_file_path, change_track_log_file_path


def read_log(file_path):
    with open(file_path) as f:
        res = [line.strip() for line in f.readlines()]
    splits = [idx for idx, line in enumerate(res) if line == '--------------------------------------------------------------------------']
    exps = {}
    for line_no in splits:
        exp = {}
        exp['run_time'] = res[line_no - 21]
        exp['config'] = {**{'shape'       : [int(i) for i in res[line_no - 17][res[line_no - 17].find(':') + 1: res[line_no - 17].find(',')].split('×')],
                          'total_place' : int(res[line_no - 17][res[line_no - 17].rfind(':') + 1:]),
                          'arrival_rate': float(res[line_no - 18][res[line_no - 18].rfind(':') + 1:res[line_no - 18].rfind(' ')]),
                          },
                         **{i.split(' ')[1]:int(i.split(' ')[0]) for i in res[line_no-16].split(', ')}
                         }
        exp['res'] = {'warehouse' : {'finish_rate': [float(r.strip('%')) / 100 for r in res[line_no - 15][15:].split(', ')],
                                     'stock_rate' : float(res[line_no - 14][36:].split(', lock_rate: ')[0].strip('%')) / 100,
                                     'lock_rate'  : float(res[line_no - 14][36:].split(', lock_rate: ')[1].strip('%')) / 100,
                                     },
                      'robots'    : {'psb_utility': {psb[:5]: float(psb[6:-1]) / 100 for psb in res[line_no - 12].replace('\'', '').lstrip('[').rstrip(']').split(', ')},
                                     'pst_utility': {pst[:5]: float(pst[6:-1]) / 100 for pst in res[line_no - 11].replace('\'', '').lstrip('[').rstrip(']').split(', ')},
                                     },
                      'ec'        : {'R-jobs'   : {'avg'  : float(res[line_no - 9][res[line_no - 9].find('avg') + 5:res[line_no - 9].find('KJ') - 1]),
                                                   'total': float(res[line_no - 9][res[line_no - 9].find('total') + 7:res[line_no - 9].rfind('KJ') - 1])},
                                     'S-jobs'   : {'avg'  : float(res[line_no - 8][res[line_no - 8].find('avg') + 5:res[line_no - 8].find('KJ') - 1]),
                                                   'total': float(res[line_no - 8][res[line_no - 8].find('total') + 7:res[line_no - 8].rfind('KJ') - 1])
                                                   },
                                     'reshuffle': float(res[line_no - 7][res[line_no - 7].find('reshuffle') + 10:res[line_no - 9].find('KJ') - 4]),
                                     'Total'    : float(res[line_no - 6][res[line_no - 6].find(':') + 2:res[line_no - 6].find('KJ') - 3])},

                      'reshuffle' : {'percent'  : float(res[line_no - 5][res[line_no - 5].find(':') + 2:][:5]) / 100,
                                     'avg_tiers': float(res[line_no - 5][res[line_no - 5].find(',') + 2:])
                                     },
                      'efficiency': {'R': {**{j[0]: float(j[1][:-1]) for j in [i.split(': ') for i in res[line_no - 3][3:].split(', ')][:2]},
                                           **{j[0]: float(j[1][:-1]) / 100 for j in [res[line_no - 3][res[line_no - 3].find('reshuffle_time part'):].split(': ')]}
                                           },
                                     'S': {**{j[0]: float(j[1][:-1]) for j in [i.split(': ') for i in res[line_no - 2][3:].split(', ')][:2]}}}
                      }
        exps[exp['run_time']] = exp
    return exps


def plot_ef_ec(experiments):
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 6), dpi=200)  # 建立R图形
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    # ax.spines['left'].set_visible(False)
    ele = {k: {'ec': v['res']['ec']['Total'], 'ef': v['res']['efficiency']['R']['mean waiting time']+v['res']['efficiency']['R']['mean working time']} for k, v in experiments.items()}
    ele_config = [v['config'] for k, v in experiments.items()]
    ec, ef = [v['ec'] for v in ele.values()], [v['ef'] for v in ele.values()]
    z = sorted(zip(ef, ec, ele_config))
    ef1, ec1, config = [i[0] for i in z], [i[1]/3600 for i in z], [i[2]['psb'] for i in z]
    plt.plot(ef1, ec1, 'o-', color='black')
    # plt.title('多轨一车效率与能耗关系', weight='bold', fontsize='x-large')
    plt.xlabel('TTO(s)', fontsize='x-large')
    plt.ylabel('ECO(kWh)', fontsize='x-large')
    for i in range(len(ec)):
        # if i == 0:
        #     config[i] = '[1tra]'
        plt.annotate(f"{config[i]} BP robots", xy=[ef1[i], ec1[i]], xytext=[ef1[i]+2, ec1[i]-0.6], fontsize='large')
    # plt.savefig('多轨一车效率能耗关系.png')
    plt.savefig('/Users/cxw/Learn/2_SIGS/Graduate/小论文/IEEECS_confs_LaTeX/IEEECS_confs_LaTeX/Fig/1BPmultiline.png')

    # plt.show()

def plot_1psb1track(experiments):
    plt.rcParams['font.sans-serif'] = 'Times New Roman'
    plt.rcParams['axes.unicode_minus'] = False

    plt.figure(figsize=(8, 6), dpi=200)  # 建立R图形
    ax = plt.axes()
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ele = {k: {'ec': v['res']['ec']['Total'], 'ef': v['res']['efficiency']['R']['mean waiting time']+v['res']['efficiency']['R']['mean working time']} for k, v in experiments.items()}
    ele_config = [v['config'] for k, v in experiments.items()]
    ec, ef = [v['ec'] for v in ele.values()], [v['ef'] for v in ele.values()]
    z = sorted(zip(ef, ec, ele_config))
    ef1, ec1, config = [i[0] for i in z], [i[1]/3600 for i in z], [i[2]['shape'][1:] for i in z]
    plt.plot(ef1, ec1, 'o-', color='black')
    # plt.title('Each track with one BP robot', weight='bold', fontsize='x-large')
    plt.xlabel('TTO(s)', fontsize='x-large')
    plt.ylabel('ECO(kWh)', fontsize='x-large')
    for i in range(len(ec)):
        plt.annotate(f"{config[i][0]}×{config[i][1]}", xy=[ef1[i], ec1[i]], xytext=[ef1[i] + 2, ec1[i]-1], fontsize='medium')
    plt.savefig('/Users/cxw/Learn/2_SIGS/Graduate/小论文/IEEECS_confs_LaTeX/IEEECS_confs_LaTeX/Fig/1BP1line.png')
    # plt.show()


if __name__ == '__main__':
    experiments = read_log(change_track_log_file_path)
    plot_ef_ec(experiments)
    # experiments = read_log(without_change_track_log_file_path)
    # plot_1psb1track(experiments)
