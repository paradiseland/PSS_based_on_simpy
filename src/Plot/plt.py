# coding=utf-8
# /usr/bin/Python3.6

"""
Author: Xingwei Chen
Email:cxw19@mails.tsinghua.edu.cn
date:2021/11/17 08:49
"""

import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import LinearLocator

from ORCSRS.Config import result_csv

pd.set_option('display.max_columns', None)
pd.set_option('max_colwidth', 200)

font_label = {
    'family': 'Times New Roman',
    'weight': 'normal',
    'usetex': True,
}
title_font_size = 18
text_font_size = 10


def plot_3d(df):
    matplotlib.rcParams['font.sans-serif'] = 'Times New Roman'
    matplotlib.rcParams['axes.unicode_minus'] = False
    # # 策略+shape的影响
    df = df.loc[(df['N_{stack}'] <= 3200) & (df['N_{sku}'] == 15) & (df['R jobs'] > 20000) & (
            df['storage policy'] != 'determined') & (df['T_w(s)'] < 1000) & (df['N_{BP}'] == 6)]
    # 策略+机器人数量的影响
    # df = df.loc[(df['shape'] == '10×40×8') & (df['N_{sku}'] == 15) & (df['R jobs'] > 20000) & (df['storage policy'] != 'determined') & (df['T_w(s)'] < 1000)]
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax1 = plt.axes(projection='3d')
    xx, yy, zz, ss = df['T_w(s)'], df['T_s(s)'], df['TEC(kJ)'], df['MR_{per}'].str.strip("%").astype(float)
    ticket_mark = df['shape'].astype(str).str.cat(df['storage policy'], sep='-')
    for x, y, z, text, s in zip(xx, yy, zz, ticket_mark, ss):
        if 'zoned' in text:
            ax1.scatter3D(x, y, z, s=s, c='#708090', marker='o')
        else:
            ax1.scatter3D(x, y, z, s=s, c='#696969', marker='o')
        ax1.text(x, y - 1.2, z, f'{text},{round(x)}s,{round(y)}s,{int(z)}kJ', fontsize='xx-small')
    # ax1.scatter3D(0, 0, 0, c='r', marker='o')
    ax1.set_zlim3d(270000, 320000)
    ax1.zaxis.set_major_locator(LinearLocator(6))
    ax1.set_xlim3d(100, 400)
    ax1.xaxis.set_major_locator(LinearLocator(4))
    ax1.set_ylim3d(140, 200)
    ax1.yaxis.set_major_locator(LinearLocator(4))
    ax1.set_xlabel('$T_{w}(s)$', fontsize='x-large', **font_label)
    ax1.set_ylabel('$T_{s}(s)$', fontsize='x-large', **font_label)
    ax1.set_zlabel('TEC(kJ)', fontsize='x-large')
    fig.tight_layout()
    # make the panes transparent
    ax1.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax1.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    # make the grid lines transparent
    # ax1.xaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax1.yaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    # ax1.zaxis._axinfo["grid"]['color'] = (1, 1, 1, 0)
    ax1.view_init(20, 200)
    # plt.show()
    plt.savefig(
        '/Users/cxw/Library/Mobile Documents/com~apple~CloudDocs/from_cxw_mac/Graduate/Code/PSS_based_on_simpy/src/Result/Figures/3d_strategy_vs_shape.png')
    # plt.savefig('/Users/cxw/Library/Mobile Documents/com~apple~CloudDocs/from_cxw_mac/Graduate/Code/PSS_based_on_simpy/src/Result/Figures/3d_strategy_vs_BP.png')


def plot_2d(df):
    matplotlib.rcParams['font.sans-serif'] = 'Times New Roman'
    matplotlib.rcParams['axes.unicode_minus'] = False
    # # 策略+shape的影响
    df = df.loc[(df['N_{stack}'] <= 3200) & (df['N_{sku}'] == 15) & (df['R jobs'] > 20000) & (
                df['storage policy'] != 'determined') & (df['T_w(s)'] < 1000) & (df['N_{BP}'] == 6)]
    df.loc[:, 'TTO(s)'] = (df['T_w(s)'].astype(float) + df['T_s(s)'].astype(float)).values.copy()
    # 策略+机器人数量的影响
    # df = df.loc[(df['shape'] == '10×40×8') & (df['N_{sku}'] == 15) & (df['R jobs'] > 20000) & (
    #         df['storage policy'] != 'determined') & (df['T_w(s)'] < 1000) & (df['TTO(s)'] < 490) & (
    #                     df['TEC(kJ)'] < 87 * 3600)]
    fig = plt.figure(figsize=(8, 6), dpi=400)
    ax1 = plt.axes()
    # xx, yy, ss = , df['TEC(kJ)'], df['MR_{per}'].str.strip("%").astype(float)
    df_zoned = df.loc[df['storage policy'] == 'zoned']
    df_random = df.loc[df['storage policy'] == 'random']
    colors = ['#708090', '#696969']
    for strategy, strategy_df, color in zip(['Zoned Storage', 'Random Storage'], [df_zoned, df_random], colors):
        strategy_df.sort_values('shape', ascending=False, inplace=True)
        strategy_df.sort_values('N_{BP}', ascending=False, inplace=True)
        ax1.plot(strategy_df['TTO(s)'], strategy_df['TEC(kJ)'] / 3600, 'o-', color=color, label=strategy)
        for x, y, text, s in zip(strategy_df['TTO(s)'], strategy_df['TEC(kJ)'] / 3600, strategy_df['shape'],
                                 strategy_df['MR_{per}'].str.strip("%").astype(float)):
            if 237 < x < 239 and 79 < y < 81:
                ax1.text(x + 3.5, y-0.3, f'{text}\n({round(x)}s, {int(y)}kJ)', fontsize=text_font_size)
            elif 256 < x < 258 and 77 < y < 79:
                ax1.text(x + 3.5, y - 0.3, f'{text}\n({round(x)}s, {int(y)}kJ)', fontsize=text_font_size)
            elif 260 < x < 262 and 75 < y < 78:
                ax1.text(x - 16, y - 0.8, f'{text}\n({round(x)}s, {int(y)}kJ)', fontsize=text_font_size)
            elif 344 < x < 346 and 78 < y < 80:
                ax1.text(x + 4, y - 0.2, f'{text}\n({round(x)}s, {int(y)}kJ)', fontsize=text_font_size)
            elif 265 < x < 267 and 85 < y < 87:
                ax1.text(x + 30, y - 0.8, f'{text}\n({round(x)}s, {int(y)}kJ)', fontsize=text_font_size)
            # if 395 < x < 400:
            #     ax1.text(x - 2, y + 1.3, f'{text}BP({round(x)}s, {int(y)}kJ)', fontsize=text_font_size)
            # elif 349 < x < 351 and 77 < y < 79:
            #     ax1.text(x + 6, y - 0.6, f'{text}BP({round(x)}s, {int(y)}kJ)', fontsize=text_font_size)
            # elif 260 < x < 262 and 75 < y < 77:
            #     ax1.text(x + 5, y - 0.6, f'{text}BP({round(x)}s, {int(y)}kJ)', fontsize=text_font_size)
            else:
                ax1.text(x + 3, y - 0.8, f'{text}\n({round(x)}s, {int(y)}kJ)', fontsize=text_font_size)
    ax1.legend(loc='upper left')
    # ax1.scatter3D(0, 0, 0, c='r', marker='o')
    # ax1.set_ylim(55, 90)
    # ax1.yaxis.set_major_locator(LinearLocator(8))
    # ax1.set_xlim(200, 500)
    # ax1.xaxis.set_major_locator(LinearLocator(4))
    ax1.set_ylim(76, 88)
    ax1.yaxis.set_major_locator(LinearLocator(7))
    ax1.set_xlim(200, 600)
    ax1.xaxis.set_major_locator(LinearLocator(5))

    ax1.set_xlabel('TTO(s)', fontsize=title_font_size)

    ax1.set_ylabel('ECO(kWh)', fontsize=title_font_size)
    # fig.tight_layout()
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    plt.show()
    # plt.savefig('/Users/cxw/Library/Mobile Documents/com~apple~CloudDocs/from_cxw_mac/Graduate/Code/PSS_based_on_simpy/src/Result/Figures/2d_strategy_vs_BP.png')
    # plt.savefig('2d_strategy_vs_shape.png', dpi=500)
    # plt.savefig('/Users/cxw/Library/Mobile Documents/com~apple~CloudDocs/from_cxw_mac/Graduate/Code/PSS_based_on_simpy/src/Result/Figures/2d_strategy_vs_BP.png')


if __name__ == '__main__':
    df = pd.read_csv(result_csv, engine='python', index_col=False)
    # plot_3d(df)
    plot_2d(df)
