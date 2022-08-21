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
    arr = list(df2[1])
    return arr


def all_data_draw(data_dict, sce_type, net_type, cmp_type, save_name):
    plt.figure()
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
    for key, lst in data_dict.items():
        # print(key, data_dict[key])
        plt.plot(range(1, len(lst) + 1, 1), lst, label=key)
        plt.xticks(range(1, len(lst) + 1, 2))

    plt.xlabel(u'网络数据集下标')
    plt.ylabel(cmp_type)
    if net_type == None:
        if sce_type == None:
            plt.title(f'mixed networks of all Scenarios')
        else:
            plt.title(u'场景 '+ str(sce_type))
    else:
        if sce_type == None:
            plt.title(f'{net_type} network of all Scenarios')
        else:
            plt.title(f'{net_type} network of Scenarios {sce_type}')

    plt.legend(loc='best')
    plt.savefig(save_name)


def cal_mean(data_dict):
    res = {}
    for key in data_dict:
        res[key] = np.mean(data_dict[key])
    return res


def get_all_data(sce_type, net_type, dir="test_log"):
    currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    files = os.listdir(f"{currentdir}/{dir}/scenario_{sce_type}/{net_type}/")
    latency_list = list(filter(lambda x: x[0] == "L", files))
    qoe_list = list(filter(lambda x: x[0] == "Q", files))
    latency_dict, qoe_dict = {}, {}
    for f in latency_list:
        latency_dict[f[8:-4]] = read_data(f"{currentdir}/{dir}/scenario_{sce_type}/{net_type}/{f}")
    for f in qoe_list:
        qoe_dict[f[4:-4]] = read_data(f"{currentdir}/{dir}/scenario_{sce_type}/{net_type}/{f}")
    return latency_dict, qoe_dict


def draw_selection():
    for sce_type in [1, 2, 3]:
        all_latency_dict, all_qoe_dict = {}, {}
        for net_type in ["low", "mid", "high"]:

            latency_dict, qoe_dict = get_all_data(sce_type, net_type, "sac_test_log")
            for k, v in qoe_dict.items():
                if k not in all_qoe_dict:
                    all_qoe_dict[k] = v
                else:
                    all_qoe_dict[k] = all_qoe_dict[k] + v
        # print("all_qoe_dict: ", all_qoe_dict)
        new_qoe_dict = {}
        for k, v in all_qoe_dict.items():
            if k in ['EDF_SAC-CC', 'DRL_TC', 'DTP_SAC-CC', 'HPF_SAC-CC']:
                new_qoe_dict[k] = v
        # for k in new_qoe_dict:
        #     print(sce_type, k, np.mean(new_qoe_dict[k]))
        #     print(k, (sum(new_qoe_dict["DRL_TC"]) - sum(new_qoe_dict[k])) / sum(new_qoe_dict[k]))
        all_data_draw(new_qoe_dict, sce_type, None, "QoE", f"pictures/scenario_{sce_type}_block_test.tiff")


def draw_cc():
    for sce_type in [1, 2, 3]:
        all_latency_dict, all_qoe_dict = {}, {}
        for net_type in ["low", "mid", "high"]:

            latency_dict, qoe_dict = get_all_data(sce_type, net_type, "sac_test_log")
            for k, v in qoe_dict.items():
                if k not in all_qoe_dict:
                    all_qoe_dict[k] = v
                else:
                    all_qoe_dict[k] = all_qoe_dict[k] + v
        # print("all_qoe_dict: ", all_qoe_dict)
        new_qoe_dict = {}
        for k, v in all_qoe_dict.items():
            if k[:3] == 'EDF':
                k = k[4:]
                new_qoe_dict[k] = v
        # for k in new_qoe_dict:
        #     print(sce_type, k, np.mean(new_qoe_dict[k]))
            # print(k, (sum(new_qoe_dict[""]) - sum(new_qoe_dict[k])) / sum(new_qoe_dict[k]))
        all_data_draw(new_qoe_dict, sce_type, None, "QoE", f"pictures/cc_scenario_{sce_type}.tiff")

if __name__ == '__main__':
    # draw_cc()
    draw_selection()
    # latency_dict_1, qoe_dict_1 = get_all_data(sce_type=1, net_type="low")
    #
    # latency_dict_2, qoe_dict_2 = get_all_data(sce_type=1, net_type="mid")
    # latency_dict_3, qoe_dict_3 = get_all_data(sce_type=1, net_type="high")

    # data_dict, sce_type, net_type, cmp_type, save_name
    # all_data_draw(latency_dict_1, 1, "low", "latency", "latency_scenario_1_low_net.png")


            # for k, v in latency_dict.items():
            #     if k not in all_qoe_dict:
            #         all_latency_dict[k] = v
            #     else:
            #         all_latency_dict[k]

            # for k in qoe_dict:
            #     print(k, np.mean(qoe_dict[k]))
            # all_data_draw(qoe_dict, sce_type, net_type, "QoE", f"pictures/qoe_scenario_{sce_type}_{net_type}_net.png")
    # all_data_draw(qoe_dict, 1, "low", "QoE", "qoe_scenario_1_low_net.png")

    # for k in qoe_dict_1:
    #     qoe_dict[k] = []
    # for k in qoe_dict:
    #     qoe_dict[k] = qoe_dict_1[k] + qoe_dict_2[k] + qoe_dict_3[k]
    # # data_dict, sce_type, net_type, cmp_type, save_name
    # # all_data_draw(qoe_dict, None, "high", "QoE", "qoe_high_net.png")
    # for k in qoe_dict:
    #     print(k, np.mean(qoe_dict[k]))


