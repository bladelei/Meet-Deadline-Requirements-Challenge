import importlib
import os, sys, inspect
import pathlib
import pandas as pd

from EDF_SAC_CC import DRLTC
from deep_agent.SAC import SAC
# from deep_agent.SAC_FC import SAC
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from simple_emulator import create_emulator, analyze_emulator, plot_cwnd, plot_rate
from simple_emulator import cal_qoe
import matplotlib.pyplot as plt


def main():
    block_traces = ["../../datasets/scenario_2/blocks/block_video.csv", "../../datasets/scenario_2/blocks/block_audio.csv"]
    network_trace = "../../datasets/scenario_2/networks/traces_108.txt"

    log_packet_file = "./output/packet_log/packet-0.log"
    second_block_file = ["../../datasets/background_traffic_traces/web.csv"]

    sac = SAC()
    # sac.load()
    my_solution = DRLTC(sac)

    # Create the emulator using your solution
    # Set second_block_file=None if you want to evaluate your solution in situation of single flow
    # Specify ENABLE_LOG to decide whether or not output the log of packets. ENABLE_LOG=True by default.
    # You can get more information about parameters at https://github.com/AItransCompetition/simple_emulator/tree/master#constant
    emulator = create_emulator(
        block_file=block_traces,
        second_block_file=None,
        trace_file=network_trace,
        solution=my_solution,
        queue_range=(20, 20),
        # enable logging packet. You can train faster if ENABLE_LOG=False
        ENABLE_LOG=True
    )

    emulator.run_for_dur(50)
    emulator.print_debug()

    print("transition_num: ", sac.get_transition())
    # sac.save()
    print()

    analyze_emulator(log_packet_file, file_range="all", sender=[1])

    plot_rate(log_packet_file, trace_file=network_trace, file_range="all", sender=[1])

    print("Qoe : %d" % (cal_qoe()))


def create_env(block_traces, network_trace, second_block_file, solution, enable_log=True):

    emulator = create_emulator(
        block_file=block_traces,
        second_block_file=second_block_file,
        trace_file=network_trace,
        solution=solution,
        queue_range=(20, 20),
        # enable logging packet. You can train faster if ENABLE_LOG=False
        ENABLE_LOG=enable_log
    )

    return emulator


def train_one_scenario():
    block_traces = ["../../datasets/scenario_2/blocks/block_video.csv",
                    "../../datasets/scenario_2/blocks/block_audio.csv"]
    network_trace = "../../datasets/scenario_2/networks/traces_7.txt"

    log_packet_file = "./output/packet_log/packet-0.log"
    second_block_file = ["../../datasets/background_traffic_traces/web.csv"]
    data_path = './'
    sac = SAC()
    my_solution = DRLTC(sac)
    record_trajectory_reward = []
    record_qoe = []
    for i in range(10):
        if i != 0: sac.load()
        sac.reset_buffer()
        my_solution.reset()
        my_solution.update_model(sac)
        emulator = create_env(block_traces, network_trace, second_block_file, my_solution)
        emulator.run_for_dur(50)
        sac.save()
        emulator.print_debug()
        qoe = cal_qoe()
        print("Qoe : %d" % (qoe))
        record_qoe.append(qoe)
        reward = my_solution.get_trajectory_reward()
        time_step = my_solution.get_time_step()
        trajectory_reward = reward / time_step
        record_trajectory_reward.append(trajectory_reward)

        if i == 9:
            print(record_trajectory_reward)
            plt.plot(range(len(record_trajectory_reward)), record_trajectory_reward)
            plt.xlabel('Trajectory')
            plt.ylabel('Averaged trajectory reward')
            plt.savefig('%sreward_record.jpg' % (data_path))


def open_train_sac():
    for test_day in [1, 2, 3]:
        BLOCK_BASE = '../../datasets/scenario_' + str(test_day) + "/blocks/"
        NETWORK_BASE = '../../datasets/scenario_' + str(test_day) + "/networks/"
        block_files = sorted([str(_) for _ in pathlib.Path(BLOCK_BASE).iterdir()])
        qoe_list, reward_list, latency_list = [], [], []
        for trace_name in sorted(os.listdir(NETWORK_BASE)):
            net_trace_file = pathlib.Path(NETWORK_BASE + trace_name)
            print("current network trace name: {}".format(trace_name))
            sac = SAC()
            sac.load()
            sac.reset_buffer()
            my_solution = DRLTC(sac)
            for i in range(10):
                sac.load()
                sac.reset_buffer()
                my_solution.reset()
                my_solution.update_model(sac)
                emulator = create_env(block_files, net_trace_file, None, my_solution)
                emulator.run_for_dur(50)
                sac.save()
                emulator.print_debug()

                qoe = cal_qoe()
                print("Qoe : %d" % (qoe))
                qoe_list.append(qoe)

                reward = my_solution.get_trajectory_reward()
                time_step = my_solution.get_time_step()
                trajectory_reward = reward / time_step
                reward_list.append(trajectory_reward)

                counter = my_solution.get_counter()
                latency = my_solution.get_trajectory_latency()
                latency_list.append(latency / counter)

            pd.DataFrame(qoe_list).to_csv('../../train_log/scenario_{0}/qoe_{1}.csv'.format(test_day, trace_name))
            pd.DataFrame(reward_list).to_csv('../../train_log/scenario_{0}/reward_{1}.csv'.format(test_day, trace_name))
            pd.DataFrame(latency_list).to_csv('../../train_log/scenario_{0}/latency_{1}.csv'.format(test_day, trace_name))
            qoe_list, reward_list, latency_list = [], [], []
            print("network trace: {} is trained".format(trace_name))


if __name__ == '__main__':
    # open_train_sac()
    main()
