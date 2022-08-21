import os, inspect
import math

import numpy as np
import random

from simple_emulator import CongestionControl
from simple_emulator import BlockSelection
from simple_emulator import cal_qoe
from simple_emulator import constant

EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'

SLOW_RATE = 1.1
CC_PERIOD = 0.01
MAX_BANDWIDTH = 900000
START_RATE = 2.4
CC_ADD_RATE = 3000
CC_MINUS_RATE = 0.01
INIT_SR = 30000.0
epsilon = 0.1
gamma = 0.9


class RL(CongestionControl):

    def __init__(self):
        super(RL, self).__init__()
        self.USE_CWND = False
        self.send_rate = INIT_SR
        # self.send_rate = float("inf")
        self.cwnd = 1

        self.ssthresh = float("inf")
        self.curr_state = "slow_start"
        self.states = ["slow_start", "congestion_avoidance", "fast_reduce", "keep_clean"]
        # the number of lost packets
        self.drop_nums = 0
        # the number of acknowledgement packets
        self.ack_nums = 0

        # current time
        self.cur_time = -1
        # the value of cwnd at last packet event
        self.last_cwnd = 0
        # the number of lost packets received at the current moment
        self.instant_drop_nums = 0
        self.miss_block_list = []

        self.flag = True
        self.time_list = []
        self.info_list = []
        self.packet_list = []
        self.trigger_time = 0
        self.period_num = 4
        self.Q_estimation = [0, 0, 0]
        self.rewards = [0 for _ in range(self.period_num)]
        self.actions = [0 for _ in range(self.period_num)]
        self.fast_reduce_flag = False
        self.last_instant_rate = 0
        self.probe_count = 0

    def cc_trigger(self, cur_time, event_info):
        event_type = event_info["event_type"]
        event_time = cur_time
        if event_time - event_info["packet_information_dict"]['Block_info']['Create_time'] > \
                event_info["packet_information_dict"]['Block_info']['Deadline']:
            if event_info["packet_information_dict"]['Block_info']["Block_id"] not in self.miss_block_list:
                self.miss_block_list.append(event_info["packet_information_dict"]['Block_info']["Block_id"])
        self.time_list.append(event_time)
        self.info_list.append(event_info['packet_information_dict'])
        self.packet_list.append(event_info)
        if event_type == EVENT_TYPE_DROP and self.flag:
            self.curr_state = self.states[1]
            self.send_rate = max(self.send_rate // SLOW_RATE, 1)
            self.flag = False
            # print("sending_rate:", self.send_rate)
        if event_time < self.trigger_time + CC_PERIOD:
            return {
                "cwnd": self.cwnd,
                "send_rate": self.send_rate
            }
        # self.trigger_time = round(event_time, 2) + CC_PERIOD
        self.trigger_time += CC_PERIOD

        if len(self.packet_list) > 0:
            rtt = 0
            avg_infly = 0
            for i in range(len(self.packet_list)):
                latency = event_info["packet_information_dict"]["Latency"]
                infly = event_info["packet_information_dict"]["Extra"]["inflight"]
                # pacing_delay = data["packet_information_dict"]["Pacing_delay"]
                # rtt += latency + pacing_delay
                rtt += latency
                avg_infly += infly
            rtt = rtt / len(self.packet_list)
            avg_infly = avg_infly / len(self.packet_list)
        else:
            rtt = 1
            avg_infly = -1

        self.cur_time = event_time
        if self.curr_state == self.states[0]:
            self.send_rate *= START_RATE
            instant_packet = list(
                filter(lambda item: self._input_list[-1]["event_time"] - item["event_time"] < CC_PERIOD,
                       self._input_list))
            instant_rate = sum([1 for data in instant_packet if data["event_type"] == 'F']) / CC_PERIOD if len(
                instant_packet) > 0 else 0
            if rtt > 0.15 and avg_infly > 200 and (instant_rate < 0.5 * self.last_instant_rate or instant_rate <= 100):
                a = 2
                self.curr_state = self.states[2]
                self.flag = False
            if self.send_rate >= MAX_BANDWIDTH:
                self.send_rate = MAX_BANDWIDTH

            return {
                "cwnd": self.cwnd,
                "send_rate": self.send_rate
            }

        elif self.curr_state == self.states[1]:
            # sum_loss_rate = sum([1 for data in self._input_list if data["event_type"] == 'D']) / len(
            #     self._input_list)
            instant_packet = list(
                filter(lambda item: self._input_list[-1]["event_time"] - item["event_time"] < CC_PERIOD,
                       self._input_list))
            instant_loss_rate = sum([1 for data in instant_packet if data["event_type"] == 'D']) / len(
                instant_packet) if len(instant_packet) > 0 else 0
            # sum_rate = sum([1 for data in self._input_list if data["event_type"] == 'F']) / CC_PERIOD if len(
            #     self._input_list) > 1 else 0
            instant_rate = sum([1 for data in instant_packet if data["event_type"] == 'F']) / CC_PERIOD if len(
                instant_packet) > 0 else 0
            instant_success_rate = sum([1 for data in instant_packet if data["event_type"] == 'F']) / len(
                instant_packet) if len(instant_packet) > 0 else 0
            reward = instant_rate * instant_success_rate
            self.rewards.pop(0)
            self.rewards.append(reward)
            Q = self.Q_estimation[self.actions[0]]
            self.Q_estimation[self.actions[0]] = Q * (1 - gamma) + reward * gamma
            if (rtt > 0.14 and avg_infly > 250 and (instant_rate < 0.5 * self.last_instant_rate or instant_rate <= 100)) or rtt > 0.2:
                a = 2
                self.fast_reduce_flag = True
                self.curr_state = self.states[2]
            elif random.random() < epsilon:
                a = random.randint(0, 2)
                self.fast_reduce_flag = False
            else:
                a = np.argmax(self.Q_estimation)
                self.fast_reduce_flag = False
            self.actions.pop(0)
            self.actions.append(a)
            if a == 0:
                self.send_rate += CC_ADD_RATE
            elif a == 1:
                self.send_rate *= 1
            else:
                if self.fast_reduce_flag:
                    self.send_rate = 1
                else:
                    self.send_rate /= 1 + CC_MINUS_RATE
            if MAX_BANDWIDTH - self.send_rate < 1.0000001:
                self.send_rate = MAX_BANDWIDTH * 0.9
            elif self.fast_reduce_flag:
                self.send_rate = 10
            elif self.send_rate - 600 < 0.00001:
                self.send_rate = 600
            # print(event_time, round(self.send_rate, 2), a, round(rtt, 3), avg_infly, round(instant_rate, 2), round(instant_loss_rate, 3))

        # reset threshhold and cwnd in fast_recovery state
        elif self.curr_state == self.states[2]:
            self.send_rate = 10
            instant_packet = list(
                filter(lambda item: self._input_list[-1]["event_time"] - item["event_time"] < CC_PERIOD,
                       self._input_list))
            instant_rate = sum([1 for data in instant_packet if data["event_type"] == 'F']) / CC_PERIOD if len(
                instant_packet) > 0 else 0
            if avg_infly < 100:
                self.curr_state = self.states[3]
            # print(event_time, round(self.send_rate, 2), round(rtt, 3), avg_infly, round(instant_rate, 2))

        else:
            if self.probe_count % 2 == 0:
                self.send_rate = 45
            else:
                self.send_rate = 110
            self.probe_count += 1
            instant_packet = list(
                filter(lambda item: self._input_list[-1]["event_time"] - item["event_time"] < CC_PERIOD,
                       self._input_list))
            instant_rate = sum([1 for data in instant_packet if data["event_type"] == 'F']) / CC_PERIOD if len(
                instant_packet) > 0 else 0
            if instant_rate > 100:
                self.curr_state = self.states[1]
            # print(event_time, round(self.send_rate, 2), round(rtt, 3), avg_infly, round(instant_rate, 2))

        self.last_instant_rate = instant_rate
        self.time_list = []
        self.info_list = []
        self.packet_list = []

        return {
            "cwnd": self.cwnd,
            "send_rate": self.send_rate
        }

    # def append_input(self, data):
    #     # if data["packet_information_dict"]['Block_info']["Block_id"] not in self.miss_block_list:
    #     self._input_list.append(data)
    #
    #     if data["event_type"] != EVENT_TYPE_TEMP:
    #         self.cc_trigger(data)
    #         return {
    #             "cwnd" : self.cwnd,
    #             "send_rate" : self.send_rate
    #         }
    #     return None


class MySolution(BlockSelection, RL):

    # def select_packet(self, cur_time, packet_queue):
    #     """
    #     The algorithm to select which packet in 'packet_queue' should be sent at time 'cur_time'.
    #     The following example is selecting packet by the create time firstly, and radio of rest life time to deadline secondly.
    #     See more at https://github.com/AItransCompetition/simple_emulator/tree/master#packet_selectionpy.
    #     :param cur_time: float
    #     :param packet_queue: the list of Packet.You can get more detail about Block in objects/packet.py
    #     :return: int
    #     """
    #     def is_better(packet):
    #         best_block_create_time = best_packet.block_info["Create_time"]
    #         packet_block_create_time = packet.block_info["Create_time"]
    #         best_block_priority = best_packet.block_info["Priority"]
    #         packet_block_priority = packet.block_info["Priority"]
    #         best_block_size = best_packet.block_info["Size"]
    #         block_size = packet.block_info["Size"]
    #         best_block_rest = (best_block_size - best_packet.offset * 1480)
    #         block_rest = (block_size - packet.offset * 1480)
    #         best_block_ddl_rest = best_packet.block_info["Deadline"] - (cur_time - best_block_create_time)
    #         block_ddl_rest = packet.block_info["Deadline"] - (cur_time - packet_block_create_time)
    #         best_block_rest_ratio = best_block_rest / best_block_size
    #         block_rest_ratio = block_rest / block_size
    #         best_block_ddl_ratio = best_block_rest / (best_block_ddl_rest + 0.01)
    #         block_ddl_radio = block_rest / (block_ddl_rest + 0.01)
    #         best_block_rest_pkt = math.ceil(best_block_rest / 1480)
    #         block_rest_pkt = math.ceil(block_rest / 1480)
    #
    #         # if packet is miss ddl
    #         if (cur_time - packet_block_create_time) >= packet.block_info["Deadline"]:
    #             return False
    #         # if best_block is miss ddl
    #         if (cur_time - best_block_create_time) >= best_packet.block_info["Deadline"]:
    #             return True
    #         # block_rest is less
    #         if best_block_rest_ratio * best_block_rest_pkt * 2 <= block_rest_ratio * block_rest_pkt:
    #             return False
    #         # # block_ddl_rest is less
    #         if best_block_ddl_rest / best_block_rest_pkt >= block_ddl_rest / block_rest_pkt * 1.5:
    #             return False
    #         # all information
    #         if best_block_ddl_ratio * best_block_rest_ratio * (1 + 0.4 * best_block_priority) > block_ddl_radio * \
    #                 block_rest_ratio * (1 + 0.4 * packet_block_priority):
    #             return True
    #         else:
    #             return False
    #     best_packet_idx = -1
    #     best_packet = None
    #     # print("queue_length =", len(packet_queue))
    #     for idx, item in enumerate(packet_queue):
    #         if best_packet is None or is_better(item) :
    #             best_packet_idx = idx
    #             best_packet = item
    #
    #     return best_packet_idx

    def on_packet_sent(self, cur_time):
        """
        The part of solution to update the states of the algorithm when sender need to send packet.
        """
        return super().on_packet_sent(cur_time)


def path_cwd(traces_dir, blocks_dir):
    traces_list = os.listdir(traces_dir)
    for i in range(len(traces_list)):
        traces_list[i] = traces_dir + '/' + traces_list[i]
    blocks_list = os.listdir(blocks_dir)
    for i in range(len(blocks_list)):
        blocks_list[i] = blocks_dir + '/' + blocks_list[i]
    return traces_list, blocks_list


# if __name__ == '__main__':
#     # trace_list = ["traces/traces_2.txt", "traces/traces_3.txt", "traces/traces_22.txt", "traces/traces_23.txt",
#     #               "traces/traces_42.txt", "traces/traces_43.txt", "traces/traces_62.txt", "traces/traces_63.txt",
#     #               "traces/traces_82.txt", "traces/traces_83.txt", "traces/traces_102.txt", "traces/traces_103.txt"]
#
#     # trace_dir_list = ["train/day_1/networks", "train/day_2/networks", "train/day_3/networks", "train/day_4/networks",
#     #                   "train/day_4/networks", "train/day_4/networks"]
#     trace_dir_list = ["train/day_4/networks", "train/day_4/networks", "train/day_4/networks"]
#     # block_dir_list = ["train/day_1/blocks", "train/day_2/blocks", "train/day_3/blocks", "train/day_4/blocks/blocks_1",
#     #                   "train/day_4/blocks/blocks_2", "train/day_4/blocks/blocks_3"]
#     block_dir_list = ["train/day_4/blocks/blocks_1", "train/day_4/blocks/blocks_2", "train/day_4/blocks/blocks_3"]
#     score_list = []
#     for i in range(len(trace_dir_list)):
#         score = 0
#         trace_list, block_list = path_cwd(trace_dir_list[i], block_dir_list[i])
#         for trace_file in trace_list:
#             # The file path of packets' log
#             log_packet_file = "output/packet_log/packet-0.log"
#
#             # Use the object you created above
#             my_solution = MySolution()
#             # Create the emulator using your solution
#             # Specify USE_CWND to decide whether or not use crowded windows. USE_CWND=True by default.
#             # Specify ENABLE_LOG to decide whether or not output the log of packets. ENABLE_LOG=True by default.
#             # You can get more information about parameters at https://github.com/AItransCompetition/simple_emulator/tree/master#constant
#             # emulator = PccEmulator(
#             #     block_file=block_list,
#             #     trace_file=trace_file,
#             #     solution=my_solution,
#             #     SEED=1,
#             #     ENABLE_LOG=False
#             # )
#
#             # Run the emulator and you can specify the time for the emualtor's running.
#             # It will run until there is no packet can sent by default.
#             # emulator.run_for_dur(20)
#
#             # print the debug information of links and senders
#             # emulator.print_debug()
#
#             # torch.save(my_solution.actor_critic, "./models/model2.pt")
#             # Output the picture of emulator-analysis.png
#             # You can get more information from https://github.com/AItransCompetition/simple_emulator/tree/master#emulator-analysispng.
#             # analyze_pcc_emulator(log_packet_file, file_range="all")
#
#             # plot_rate(log_packet_file, trace_file="traces/trace.txt", file_range="all")
#             print(trace_file)
#             score += cal_qoe()
#             print(cal_qoe())
#         score /= 2
#         print("sum:", score)
#         print()
#         score_list.append(score)
#     print(score_list, sum(score_list))
# if __name__ == '__main__':
#     traces_list, blocks_list = path_cwd("train/day_1/networks", "train/day_1/blocks")
#     qoe_sum = 0
#
#     for trace_file in traces_list:
#         # The file path of packets' log
#         log_packet_file = "output/packet_log/packet-0.log"
#         # Use the object you created above
#         my_solution = MySolution()
#         # Create the emulator using your solution
#         # Specify USE_CWND to decide whether or not use crowded windows. USE_CWND=True by default.
#         # Specify ENABLE_LOG to decide whether or not output the log of packets. ENABLE_LOG=True by default.
#         emulator = PccEmulator(
#             block_file=blocks_list,
#             trace_file=trace_file,
#             solution=my_solution,
#             SEED=1,
#             ENABLE_LOG=False
#         )
#         # Run the emulator and you can specify the time for the emualtor's running.
#         # It will run until there is no packet can sent by default.
#         emulator.run_for_dur(20)
#         # print the debug information of links and senders
#         # emulator.print_debug()
#         # torch.save(my_solution.actor_critic, "./models/model2.pt")
#         # analyze_pcc_emulator(log_packet_file, file_range="all")
#         # plot_rate(log_packet_file, trace_file="traces/trace.txt", file_range="all")
#         print(cal_qoe())
#         qoe_sum += cal_qoe()
#
#     print(qoe_sum/2)




