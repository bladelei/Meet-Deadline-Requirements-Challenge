import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import os, sys, inspect


def read_data(file_name):
    df2 = pd.read_csv(file_name,
                      header=None,  # 默认不读取列索引
                      skiprows=1,
                      index_col=0)
    arr = np.array(df2[1])
    return arr


def all_data_draw(data_dict, save_name, day=None):
    plt.figure()
    for key in data_dict:
        res = stats.relfreq(data_dict[key], numbins=40)
        # print("res:", res)

        x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
        y = np.cumsum(res.frequency)
        plt.plot(x, y, linestyle="--", label=key)
    plt.xlabel('Average QoE')
    if day == None:
        plt.title('All Scenarios')
    else:
        plt.title(f'Scenarios {day}')
    plt.legend()
    plt.savefig(save_name)


def cal_mean(data_dict):
    res = {}
    for key in data_dict:
        res[key] = np.mean(data_dict[key])
    return res


def get_scenario_data(day, type):
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    files = os.listdir(f"{currentdir}/running_log/{type}_scenario_{day}/")
    data_dict = {}
    for f in files:
        data_dict[f] = read_data(f"{currentdir}/running_log/{type}_scenario_{day}/{f}")
    return data_dict


# def scenario_data_draw(type, save_name):
#     plt.figure(figsize=(3*3, 3))
#     for day in range(1, 4):
#         data_dict = get_scenario_data(day, type)
#         ax = plt.subplot(1, 3, day)
#         for key in data_dict:
#             res = stats.relfreq(data_dict[key], numbins=30)
#             x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
#             y = np.cumsum(res.frequency)
#             print(x, y)
#             ax.plot(x, y, linestyle="--", label=key)
#         ax.set_aspect('equal')
#         # ax.title(f'{day}Scenarios')
#         # ax.xlabel('Average QoE')
#
#     # plt.legend()
#     plt.savefig(save_name, bbox_inches='tight')


def get_all_data(type):
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    files = os.listdir(f"{currentdir}/running_log/{type}/")
    data_dict = {}
    for f in files:
        data_dict[f] = read_data(f"{currentdir}/running_log/{type}/{f}")
    return data_dict


if __name__ == '__main__':
    data_dict = get_all_data("Hybrid")
    data_dict = cal_mean(data_dict)
    print(cal_mean(data_dict))
    for k in  data_dict:
        print(data_dict[k])
        print(k, (data_dict['DRL_TC.csv'] - data_dict[k]) / data_dict[k])
    # all_data_draw(data_dict, "Hybrid.png")

    # for day in range(1, 4):
    #     data_dict = get_all_data(f"D_scenario_{day}")
    #     data_dict = cal_mean(data_dict)
    #     print(data_dict)
    #     print((data_dict['DEF_DRL_CC.csv'] - data_dict['DEF_BBR.csv']) / data_dict['DEF_BBR.csv'])
        # all_data_draw(data_dict, f"H_scenario_{day}.png", day)

    # scenario_data_draw('D', 'DEF-scenario-CC.png')
    # pass




