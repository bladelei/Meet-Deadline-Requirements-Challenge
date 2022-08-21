# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt

def draw_cc():
    name_list = ['场景 1', '场景 2', '场景 3']
    num_list = [91.65, 293.23, 195.77]
    num_list1 = [598.39, 376.69, 702.56]
    num_list2 = [555.13, 377.23, 734.75]
    num_list3 = [563.78, 413.66, 1043.14]
    x = list(range(len(num_list)))
    total_width, n = 0.6, 3
    width = total_width / n
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.bar(x, num_list, width=width, label='NewReno', fc='blue')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list1, width=width, label='BBR', tick_label=name_list, fc='yellow')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list2, width=width, label='Fast-TCP', fc='green')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list3, width=width, label='SAC-CC', fc='red')

    plt.ylabel('Average QoE')
    plt.legend()
    plt.savefig("pictures/cc_result.tiff")
    plt.show()


def draw_block_selection():
    name_list = ['场景 1', '场景 2', '场景 3']
    num_list = [563.78, 413.66, 1043.14]
    num_list1 = [440.96, 532.95, 1187.59]
    num_list2 = [624.44, 629.07, 1200.91]
    num_list3 = [667.65, 661.99, 1246.89]
    x = list(range(len(num_list)))
    total_width, n = 0.6, 3
    width = total_width / n
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS']
    plt.bar(x, num_list, width=width, label='EDF', fc='blue')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list1, width=width, label='DTP', tick_label=name_list, fc='yellow')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list2, width=width, label='HPF', fc='green')
    for i in range(len(x)):
        x[i] = x[i] + width
    plt.bar(x, num_list3, width=width, label='DRL-TC', fc='red')

    plt.ylabel('Average QoE')
    plt.legend()
    plt.savefig("pictures/block_selection.tiff")
    plt.show()


if __name__ == '__main__':
    draw_cc()
    draw_block_selection()
