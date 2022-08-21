
import importlib
import os, sys, inspect
import pathlib
import pandas as pd

from EDF_SAC_CC import DRLTC
from deep_agent.SAC import SAC
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from simple_emulator import create_emulator, analyze_emulator, plot_cwnd, plot_rate
from simple_emulator import cal_qoe
import matplotlib.pyplot as plt


def create_env(block_traces, network_trace, solution, enable_log=True):

    emulator = create_emulator(
        block_file=block_traces,
        second_block_file=None,
        trace_file=network_trace,
        solution=solution,
        queue_range=(20, 20),
        # enable logging packet. You can train faster if ENABLE_LOG=False
        ENABLE_LOG=enable_log
    )

    return emulator



def get_network_trace(type):
    NETWORK_BASE = '../../datasets/network/'
    net_dict = {"low": [], "mid": [], "high": []}

    for trace_name in sorted(os.listdir(NETWORK_BASE)):
        if 0 < int(trace_name[trace_name.find('_') + 1 : trace_name.find('.')]) <= 40:
            net_dict["low"].append(pathlib.Path(NETWORK_BASE + trace_name))
        elif 40 < int(trace_name[trace_name.find('_') + 1 : trace_name.find('.')]) <= 80:
            net_dict["mid"].append(pathlib.Path(NETWORK_BASE + trace_name))
        else:
            net_dict["high"].append(pathlib.Path(NETWORK_BASE + trace_name))

    # for k, v in net_dict.items():
    #     net_dict[k] = sorted(v, key=lambda x: int(str(x)[str(x).find('_') + 1 : str(x).find('.')]))

    return net_dict[type]


def solution_test(log_name):
    for test_day in [1, 2, 3]:
        BLOCK_BASE = '../../datasets/scenario_' + str(test_day) + "/blocks/"
        net_type = ["low", "mid", "high"]
        for type in net_type:
            qoe_list = []
            latency_list = []
            network_traces = get_network_trace(type)
            for network_trace in network_traces:
                sac = SAC()
                sac.load()
                my_solution = DRLTC(sac)
                block_files = sorted([str(_) for _ in pathlib.Path(BLOCK_BASE).iterdir()])
                emulator = create_env(block_files, network_trace, my_solution)
                emulator.run_for_dur(50)
                sac.save()
                qoe = cal_qoe(run_dir=currentdir)
                qoe_list.append(qoe)
                counter = my_solution.get_counter()
                trajectory_latency = my_solution.get_trajectory_latency()
                latency_list.append(trajectory_latency / counter)

            pd.DataFrame(qoe_list).to_csv(f'../../test_log/scenario_{test_day}/{type}/QoE_{log_name}.csv')
            pd.DataFrame(latency_list).to_csv(f'../../test_log/scenario_{test_day}/{type}/Latency_{log_name}.csv')



if __name__ == '__main__':
    # get_network_trace("mid")
    solution_test("DRL-TC")