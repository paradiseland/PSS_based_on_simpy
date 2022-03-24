# coding=utf-8
# /usr/bin/Python3.6


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def find_x():
    x = [10000] + np.random.normal(loc=1.001, scale=0.005, size=500).tolist() + np.random.normal(1.00005, 0.0005,
                                                                                                 size=500).tolist()
    xx = np.cumprod(x)
    np.save("../Result/rewards.npy", xx)
    plt.plot(xx)
    plt.show()


def plot():
    mean_len = 50
    rc = {'axes.unicode_minus': False}
    sns.set(style='ticks', font='SimHei', rc=rc)
    x = np.load("../Result/rewards-9.npy")
    x = x/10000*13257-30000
    xx = [np.mean(x[i * mean_len: i * mean_len + mean_len]) for i in range(int(1000 / mean_len))]
    reward = plt.plot(x, linewidth=1.1, label='回合奖励')
    ma_rewards = plt.plot([i * mean_len for i in range(int(x.size / mean_len))], xx, linewidth=0.8, linestyle='-.',
                          label=f'每{mean_len}回合平均奖励')
    plt.legend(loc='best')
    plt.yticks([-16000, -14000, -12000, -10000], [-16000, -14000, -12000, -10000])
    plt.xlabel('回合')
    plt.ylabel('奖励')
    plt.grid(True)
    dpi=500
    plt.savefig(f'../Result/Figures/rewards-{dpi}dpi.png', dpi=dpi)


if __name__ == '__main__':
    plot()
