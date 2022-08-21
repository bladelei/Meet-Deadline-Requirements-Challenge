"""
This demo aims to help player running system quickly by using the pypi library simple-emualtor https://pypi.org/project/simple-emulator/.
"""
from simple_emulator import CongestionControl
from simple_emulator import Reno

# We provided a simple algorithms about packet selection to help you being familiar with this competition.
# In this example, it will select the packet according to packet's created time first and radio of rest life time to deadline secondly.
from simple_emulator import BlockSelection
# import SAC_RL

import numpy as np;

# for tf version < 2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random


# change every ACKS times
ACKS = 10
#throughput * 2, lost rate *2, len(status_episode[0]) * ACKS
state_dim = 2 + 2 + ACKS * 8

EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'

MAX_BANDWITH = 25000
MIN_BANDWITH = 500


np.random.seed(2)
torch.manual_seed(1)

EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'

class DRLTC(Reno):

    def __init__(self, sac):
        super(DRLTC, self).__init__()
        self.USE_CWND = False
        self.sac = sac

        # self.USE_CWND=False
        self.send_rate = 500.0

        # list to store the input of "cc_trigger"
        self._input_list = []
        self.rtt = 0

        self.max_send_rate = 500.0
        self.min_latency = 100.0

        self.counter = 0  # EPISODE counter
        self.time_step = 0
        self.trajectory_reward = 0
        self.trajectory_latency = 0
        self.random_counter = 100
        self.Lambda = 0.9

        # block_dict[0]记录pak所属的block已经完成的包数，[1]记录丢包信息
        self.block_dict = {}
        # pk的type，0为ACK,1为drop
        self.type_list = []
        self.priority_list = []
        self.left_time_list = []
        self.left_num_list = []
        # state_episode记录每一个ack收集到的重要信息
        self.state_episode = []
        self.last_state = np.zeros(state_dim)
        self.last_action = 2

        self.event_nums = 0
        self.event_lost_nums = 0
        self.event_ack_nums = 0

    def reset(self):
        # self.USE_CWND=False
        self.send_rate = 500.0

        # list to store the input of "cc_trigger"
        self._input_list = []
        self.rtt = 0

        self.max_send_rate = 500.0
        self.min_latency = 100.0

        self.counter = 0  # EPISODE counter
        self.time_step = 0
        self.trajectory_reward = 0
        self.trajectory_latency = 0
        self.random_counter = 100
        self.Lambda = 0.9

        # block_dict[0]记录pak所属的block已经完成的包数，[1]记录丢包信息
        self.block_dict = {}
        # pk的type，0为ACK,1为drop
        self.type_list = []
        self.priority_list = []
        self.left_time_list = []
        self.left_num_list = []
        # state_episode记录每一个ack收集到的重要信息
        self.state_episode = []
        self.last_state = np.zeros(state_dim)
        self.last_action = 2

        self.event_nums = 0
        self.event_lost_nums = 0
        self.event_ack_nums = 0

    def select_block(self, cur_time, block_queue):
        '''
        The alogrithm to select the block which will be sended in next.
        The following example is selecting block by the create time firstly, and radio of rest life time to deadline secondly.
        :param cur_time: float
        :param block_queue: the list of Block.You can get more detail about Block in objects/block.py
        :return: int
        '''

        def is_better(block):
            best_block_create_time = best_block.block_info["Create_time"]
            cur_block_create_time = block.block_info["Create_time"]
            # if block is miss ddl
            if (cur_time - cur_block_create_time) >= block.block_info["Deadline"]:
                return False
            if (cur_time - best_block_create_time) >= best_block.block_info["Deadline"]:
                return True
            if best_block_create_time != cur_block_create_time:
                return best_block_create_time > cur_block_create_time
            return (cur_time - best_block_create_time) * best_block.block_info["Deadline"] > \
                   (cur_time - cur_block_create_time) * block.block_info["Deadline"]

        best_block_idx = -1
        best_block = None
        for idx, item in enumerate(block_queue):
            if best_block is None or is_better(item):
                best_block_idx = idx
                best_block = item

        return best_block_idx


    def on_packet_sent(self, cur_time):
        """
        The part of solution to update the states of the algorithm, when sender need to send packet.
        """
        return super().on_packet_sent(cur_time)

    def estimate_bandwidth(self, cur_time, event_info):
        # append the input, help the estimator to do long-term decisions
        self._input_list.append([cur_time, event_info])

        event_type = event_info["event_type"]
        event_time = cur_time
        packet = event_info["packet_information_dict"]
        block = packet["Block_info"]
        event_block_id = block["Block_id"]

        block_priority = 3 - block["Priority"]
        block_ddl = block["Deadline"]
        block_create_time = block["Create_time"]
        block_split_nums = block["Split_nums"]

        inflight = packet["Extra"]["inflight"]
        latency = packet["Latency"] + packet["Send_delay"] + packet["Pacing_delay"]
        packet_id = packet["Packet_id"]
        block_res_time = block_ddl + block_create_time - event_time
        # block_res_nums = block_split_nums - packet_idx

        self.event_nums += 1
        if event_block_id not in self.block_dict:
            self.block_dict[event_block_id] = [0, []]

        if event_type == EVENT_TYPE_DROP:
            self.type_list.append(1)
            self.block_dict[event_block_id][1].append(packet_id)
            self.event_lost_nums += 1
        else:
            self.block_dict[event_block_id][0] += 1
            if packet_id in self.block_dict[event_block_id][1]:
                self.block_dict[event_block_id][1].remove(packet_id)
            self.type_list.append(0)
            self.event_ack_nums += 1
            # update rtt
        block_res_nums = block_split_nums - self.block_dict[event_block_id][0]
        # print("block_res_nums: ", block_res_nums)
        pac_exp_rate = block_res_nums / block_res_time if block_res_time > 0 else 0

        alf = 0.9
        self.rtt = alf * latency + (1 - alf) * self.rtt
        self.max_send_rate = max(self.send_rate, self.max_send_rate)
        self.min_latency = min(latency, self.min_latency)

        self.priority_list.append(block_priority)
        self.left_time_list.append(block_res_time)
        self.left_num_list.append(block_res_nums)

        info = []
        info.append(self.send_rate / MAX_BANDWITH)
        info.append(latency)
        info.append(self.max_send_rate / MAX_BANDWITH)
        info.append(self.min_latency)
        info.append(inflight)
        info.append(self.cwnd)
        info.append(pac_exp_rate)
        info.append(len(self.block_dict[event_block_id][1]))
        self.state_episode.append(info)

        self.counter += 1
        self.trajectory_latency += latency

        if self.counter % ACKS == 0:  # choose action every ACKS times

            # print("ACKS: send_rate is {}".format(self.send_rate))

            sum_loss_rate = self.event_lost_nums / self.event_nums
            sum_rate = self.event_ack_nums / event_time

            # 10ms 内的丢包率
            instant_packet = list(filter(lambda item: self._input_list[-1][0] - item[0] < 0.01, self._input_list))
            self._input_list[:] = instant_packet
            instant_loss_nums = sum([1 for data in instant_packet if data[1]["event_type"] == 'D'])
            instant_loss_rate = instant_loss_nums / len(instant_packet) if len(instant_packet) > 0 else 0


            # 10ms 内的吞吐速率
            instant_ack_packet = list(filter(lambda data: data[1]["event_type"] == 'F', instant_packet))
            if (instant_packet[-1][0] - instant_packet[0][0]) > 0:
                instant_rate = len(instant_ack_packet) / (instant_packet[-1][0] - instant_packet[0][0])
            else:
                instant_rate = 0


            # current reward
            r = 0
            for i in range(ACKS):
                if self.type_list[i] == 0:
                    if self.left_num_list[i] == 0 and self.left_time_list[i] >= 0:
                        r += 1
                    r += self.send_rate / MAX_BANDWITH
                else:
                    r -= abs(self.priority_list[i] * 0.5 * self.left_time_list[i])

            # print("reward of {}: {}".format(self.counter, r))
            self.trajectory_reward += r
            self.time_step += 1
            # current status
            s_ = []

            for i in range(ACKS):
                for j in range(len(self.state_episode[0])):
                    s_.append(float(self.state_episode[i][j]))

            s_.append(float(sum_loss_rate))
            s_.append(float(instant_loss_rate))
            s_.append(float(sum_rate))
            s_.append(float(instant_rate))

            s_array = np.array(s_)

            # store
            self.sac.store(self.last_state, self.last_action, r, s_array)
            # self.dqn.store_transition(self.last_state, self.last_action, r, s_array)

            # choose action
            # a = self.dqn.choose_action(s_array)
            a = self.sac.select_action(s_array)
            self.last_action = a

            # exploration
            # declining random rate
            # self.random_counter -= 1
            # if self.random_counter <= 0:
            #     self.Lambda /= 2.0
            #     if self.Lambda < 0.05:
            #         self.Lambda = 0.05
            #     # self.random_counter = 50
            # if random.random() < self.Lambda:
            #     a = random.uniform(-1.0, 1.0)

            self.send_rate = (1.5 + a) * self.send_rate
            # self.send_rate = (3 ** a) * self.send_rate

            if self.send_rate < MIN_BANDWITH:
                self.send_rate = MIN_BANDWITH
            if self.send_rate > MAX_BANDWITH:
                self.send_rate = MAX_BANDWITH


            self.last_state = s_
            self.type_list = []
            self.priority_list = []
            self.left_time_list = []
            self.left_num_list = []
            self.state_episode = []

            if self.sac.num_transition >= 300:
                self.sac.update()


    def cc_trigger(self, cur_time, event_info):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        self.estimate_bandwidth(cur_time, event_info)

        # set cwnd or sending rate in sender according to bandwidth estimator
        return {
            "cwnd" : self.cwnd,
            "send_rate" : self.send_rate,
        }

    def update_model(self, sac):
        self.sac = sac

    def get_trajectory_reward(self):

        return self.trajectory_reward

    def get_trajectory_latency(self):

        return self.trajectory_latency

    def get_counter(self):

        return self.counter

    def get_time_step(self):

        return self.time_step

    def get_last_state(self):

        return self.last_state

    def get_last_action(self):

        return self.last_action

