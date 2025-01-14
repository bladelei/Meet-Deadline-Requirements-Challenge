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
        x = res.lowerlimit + np.linspace(0, res.binsize * res.frequency.size, res.frequency.size)
        y = np.cumsum(res.frequency)
        plt.plot(x, y, linestyle="--", label=key)
    plt.xlabel('Average QoE')
    plt.ylabel('CDF')
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
        data_dict[f[:-4]] = read_data(f"{currentdir}/running_log/{type}/{f}")
    return data_dict


if __name__ == '__main__':

    # #EDF
    # data_dict = get_all_data("EDF")
    # all_data_draw(data_dict, "EDF.png")
    #
    # mean_data_dict = cal_mean(data_dict)
    # print("mean_data_dict: ",mean_data_dict)
    # for k in mean_data_dict:
    #     # print(k)
    #     print(k, mean_data_dict[k])
    #     print(k, (mean_data_dict['EDF_DRL_CC'] - mean_data_dict[k]) / mean_data_dict[k])
    #
    # for day in range(1, 4):
    #     data_dict = get_all_data(f"E_scenario_{day}")
    #     all_data_draw(data_dict, f"E_scenario_{day}.png", day)
    #     mean_data_dict = cal_mean(data_dict)
    #     print(f"E_scenario_mean_data_dict_{day}: ", mean_data_dict)
    #     for k in mean_data_dict: print(k, (mean_data_dict['EDF_DRL_CC'] - mean_data_dict[k]) / mean_data_dict[k])



    # #Select
    data_dict = get_all_data("Select")
    print(data_dict)
    # all_data_draw(data_dict, "Select.png")
    #
    # mean_data_dict = cal_mean(data_dict)
    # print("mean_data_dict: ",mean_data_dict)
    # for k in mean_data_dict:
    #     # print(k)
    #     print(k, mean_data_dict[k])
    #     print(k, (mean_data_dict['DRL_TC'] - mean_data_dict[k]) / mean_data_dict[k])
    # for day in range(1, 4):
    #     data_dict = get_all_data(f"S_scenario_{day}")
    #     all_data_draw(data_dict, f"S_scenario_{day}.png", day)
    #     mean_data_dict = cal_mean(data_dict)
    #     print(f"S_scenario_mean_data_dict_{day}: ", mean_data_dict)
    #     for k in mean_data_dict:
    #         print(k, (mean_data_dict['DRL_TC'] - mean_data_dict[k]) / mean_data_dict[k])


    # #Hybrid
    # data_dict = get_all_data("Hybrid")
    # all_data_draw(data_dict, "Hybrid.png")
    #
    # mean_data_dict = cal_mean(data_dict)
    # print("mean_data_dict: ",mean_data_dict)
    # for k in mean_data_dict:
    #     # print(k)
    #     print(k, mean_data_dict[k])
    #     print(k, (mean_data_dict['DRL_TC'] - mean_data_dict[k]) / mean_data_dict[k])
    #
    # for day in range(1, 4):
    #     data_dict = get_all_data(f"H_scenario_{day}")
    #     all_data_draw(data_dict, f"H_scenario_{day}.png", day)
    #     mean_data_dict = cal_mean(data_dict)
    #     print(f"H_scenario_mean_data_dict_{day}: ", mean_data_dict)
    #     for k in mean_data_dict: print(k, (mean_data_dict['DRL_TC'] - mean_data_dict[k]) / mean_data_dict[k])






