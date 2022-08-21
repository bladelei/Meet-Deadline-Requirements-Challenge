
import importlib
import os, sys, inspect
import pathlib
import pandas as pd



# from deep_agent.SAC import SAC
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from simple_emulator import create_emulator, analyze_emulator, plot_cwnd, plot_rate
from simple_emulator import cal_qoe
import matplotlib.pyplot as plt
# from solutions.sac_tc.EDF_SAC_CC import DRLTC
from solutions.sac_tc.deep_agent.SAC import SAC


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


def get_file_name(key):
    # EDF
    EDF_NewReno = 'solutions.reno.EDF-NewReno'
    EDF_Fast_TCP = 'solutions.Fast-TCP.EDF-Fast-TCP'
    EDF_BBR = 'solutions.bbr.EDF-BBR'
    # EDF_DRL_CC = 'solutions.dqn.EDF-DRL-CC'
    EDF_CUBIC = 'solutions.cubic.EDF-CUBIC'
    EDF_COPA = 'solutions.copa.EDF-COPA'

    #Hybrid
    DTP_Fast_TCP = 'solutions.Fast-TCP.DTP-Fast-TCP'
    HPF_BBR = 'solutions.bbr.HPF-BBR'
    DRL_TC = 'solutions.dqn.DRL-TC'

    #Select
    EDF_DRL_CC = 'solutions.dqn.EDF-DRL-CC'
    HPF_DRL_CC = 'solutions.dqn.HPF-DRL-CC'
    DTP_DRL_CC = 'solutions.dqn.DTP-DRL-CC'

    d = {EDF_NewReno: "EDF_NewReno",
         EDF_Fast_TCP: "EDF_Fast_TCP",
         EDF_BBR: "EDF_BBR",
         EDF_DRL_CC: "EDF_DRL_CC",
         EDF_CUBIC: "EDF_CUBIC",
         EDF_COPA: 'EDF_COPA',
         DTP_Fast_TCP: "DTP_Fast_TCP",
         HPF_BBR: "HPF_BBR",
         DRL_TC: "DRL_TC",
         HPF_DRL_CC: "HPF_DRL_CC",
         DTP_DRL_CC: "DTP_DRL_CC"}

    return d[key]


def get_network_trace(type):
    NETWORK_BASE = 'datasets/network/'
    net_dict = {"low": [], "mid": [], "high": []}

    for trace_name in sorted(os.listdir(NETWORK_BASE)):
        if 0 < int(trace_name[trace_name.find('_') + 1 : trace_name.find('.')]) <= 40:
            net_dict["low"].append(pathlib.Path(NETWORK_BASE + trace_name))
        elif 40 < int(trace_name[trace_name.find('_') + 1 : trace_name.find('.')]) <= 80:
            net_dict["mid"].append(pathlib.Path(NETWORK_BASE + trace_name))
        else:
            net_dict["high"].append(pathlib.Path(NETWORK_BASE + trace_name))

    for k, v in net_dict.items():
        net_dict[k] = sorted(v, key=lambda x: int(str(x)[str(x).find('_') + 1 : str(x).find('.')]))

    return net_dict[type]


def solution_test(solution_file, log_name):
    for test_day in [1, 2, 3]:
        BLOCK_BASE = 'datasets/scenario_' + str(test_day) + "/blocks/"
        net_type = ["low", "mid", "high"]
        for type in net_type:
            qoe_list = []
            latency_list = []
            network_traces = get_network_trace(type)
            for network_trace in network_traces:
                block_files = sorted([str(_) for _ in pathlib.Path(BLOCK_BASE).iterdir()])
                solution = importlib.import_module(solution_file)
                my_solution = solution.MySolution()
                emulator = create_env(block_files, network_trace, my_solution)
                emulator.run_for_dur(50)
                qoe = cal_qoe(run_dir=currentdir)
                qoe_list.append(qoe)
                counter = my_solution.get_counter()
                trajectory_latency = my_solution.get_trajectory_latency()
                latency_list.append(trajectory_latency / counter)

            pd.DataFrame(qoe_list).to_csv(f'test_log/scenario_{test_day}/{type}/QoE_{log_name}.csv')
            pd.DataFrame(latency_list).to_csv(f'test_log/scenario_{test_day}/{type}/Latency_{log_name}.csv')


# def sac_test(log_name):
#     sac = SAC()
#     sac.load()
#     my_solution = DRLTC(sac)
#     for test_day in [1, 2, 3]:
#         BLOCK_BASE = 'datasets/scenario_' + str(test_day) + "/blocks/"
#         net_type = ["low", "mid", "high"]
#         for type in net_type:
#             qoe_list = []
#             latency_list = []
#             network_traces = get_network_trace(type)
#             for network_trace in network_traces:
#                 block_files = sorted([str(_) for _ in pathlib.Path(BLOCK_BASE).iterdir()])
#                 sac.reset_buffer()
#                 my_solution.reset()
#                 my_solution.update_model(sac)
#                 emulator = create_env(block_files, network_trace, my_solution)
#                 emulator.run_for_dur(50)
#                 qoe = cal_qoe(run_dir=currentdir)
#                 qoe_list.append(qoe)
#                 counter = my_solution.get_counter()
#                 trajectory_latency = my_solution.get_trajectory_latency()
#                 latency_list.append(trajectory_latency / counter)
#
#             pd.DataFrame(qoe_list).to_csv(f'test_log/scenario_{test_day}/{type}/QoE_{log_name}.csv')
#             pd.DataFrame(latency_list).to_csv(f'test_log/scenario_{test_day}/{type}/Latency_{log_name}.csv')

def sac_test_without_module(solution_file, log_name):

    for test_day in [1, 2, 3]:
        BLOCK_BASE = 'datasets/scenario_' + str(test_day) + "/blocks/"
        net_type = ["low", "mid", "high"]
        sac = SAC()
        for type in net_type:
            qoe_list = []
            latency_list = []
            network_traces = get_network_trace(type)
            for network_trace in network_traces:
                block_files = sorted([str(_) for _ in pathlib.Path(BLOCK_BASE).iterdir()])
                sac.reset_buffer()
                solution = importlib.import_module(solution_file)
                my_solution = solution.DRLTC(sac)
                # my_solution = DRLTC(sac)
                emulator = create_env(block_files, network_trace, my_solution)
                emulator.run_for_dur(50)
                qoe = cal_qoe(run_dir=currentdir)
                qoe_list.append(qoe)
                counter = my_solution.get_counter()
                trajectory_latency = my_solution.get_trajectory_latency()
                latency_list.append(trajectory_latency / counter)

            pd.DataFrame(qoe_list).to_csv(f'sac_test_log/scenario_{test_day}/{type}/QoE_{log_name}.csv')
            pd.DataFrame(latency_list).to_csv(f'sac_test_log/scenario_{test_day}/{type}/Latency_{log_name}.csv')


if __name__ == '__main__':
    EDF_NewReno = 'solutions.reno.EDF-NewReno'
    EDF_Fast_TCP = 'solutions.Fast-TCP.EDF-Fast-TCP'
    EDF_BBR = 'solutions.bbr.EDF-BBR'
    EDF_DQN_CC = 'solutions.dqn.EDF_DQN_CC'
    EDF_CUBIC = 'solutions.cubic.EDF-CUBIC'
    EDF_COPA = 'solutions.copa.EDF-COPA'

    HPF_SAC_CC = 'solutions.sac_tc.HPF_SAC_CC'
    DTP_SAC_CC = 'solutions.sac_tc.DTP_SAC_CC'
    DRL_TC = 'solutions.sac_tc.DRL-TC'
    # solution_test(EDF_DQN_CC, "EDF_DQN-CC")
    # sac_test("EDF_DRL-TC")
    sac_test_without_module(HPF_SAC_CC, "HPF_SAC-CC")
    sac_test_without_module(DTP_SAC_CC, "DTP_SAC-CC")
    sac_test_without_module(DRL_TC, "DRL_TC")