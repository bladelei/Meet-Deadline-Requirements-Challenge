"""
This demo aims to help player running system quickly by using the pypi library simple-emualtor https://pypi.org/project/simple-emulator/.
"""
from simple_emulator import CongestionControl

# We provided a simple algorithms about packet selection to help you being familiar with this competition.
# In this example, it will select the packet according to packet's created time first and radio of rest life time to deadline secondly.
from simple_emulator import BlockSelection

import numpy as np;

# for tf version < 2.0
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import random

# from dueling_model import DuelingDQN

from .dueling_model import DuelingDQN

np.random.seed(2)
torch.manual_seed(1)

EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'



class DQN(object):
    def __init__(self,
                 N_STATES=10,
                 N_ACTIONS=5,
                 LR=0.01,
                 GAMMA=0.9,
                 TARGET_REPLACE_ITER=100,
                 MEMORY_CAPACITY=200,
                 BATCH_SIZE=32
                 ):

        self.N_STATES = N_STATES
        self.N_ACTIONS = N_ACTIONS
        self.LR = LR
        # self.EPSILON = EPSILON
        self.GAMMA = GAMMA
        self.TARGET_REPLACE_ITER = TARGET_REPLACE_ITER
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.BATCH_SIZE = BATCH_SIZE

        self.eval_net = DuelingDQN(self.N_STATES, self.N_ACTIONS)
        self.target_net = DuelingDQN(self.N_STATES, self.N_ACTIONS)

        self.learn_step_counter = 0  # For target update timing
        self.memory_counter = 0  # For recording old states and reward
        self.memory = np.zeros((self.MEMORY_CAPACITY, self.N_STATES * 2 + 2))  # Initial size of memory
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)  # Torch's optimizer
        self.loss_func = nn.MSELoss()  # Error formula


    def choose_action(self, x):
        x = Variable(torch.unsqueeze(torch.FloatTensor(x), 0))
        # Only enter one sample here
        actions_value = self.eval_net.forward(x)
        action = torch.max(actions_value, 1)[1].data.numpy()[0]  # return the argmax
        return action


    def store_transition(self, s, a, r, s_):
        transition = np.hstack((s, [a, r], s_))
        # shape of transition = 112
        # If the memory bank is full, overwrite the old data
        index = self.memory_counter % self.MEMORY_CAPACITY
        self.memory[index, :] = transition
        self.memory_counter += 1


    def learn(self):
        # Parameter update of target net 
        if self.learn_step_counter % self.TARGET_REPLACE_ITER == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        # Extract batch data from memory
        sample_index = np.random.choice(self.MEMORY_CAPACITY, self.BATCH_SIZE)
        b_memory = self.memory[sample_index, :]
        b_s = Variable(torch.FloatTensor(b_memory[:, :self.N_STATES]))
        b_a = Variable(torch.LongTensor(b_memory[:, self.N_STATES:self.N_STATES+1].astype(int)))
        b_r = Variable(torch.FloatTensor(b_memory[:, self.N_STATES+1:self.N_STATES+2]))
        b_s_ = Variable(torch.FloatTensor(b_memory[:, -self.N_STATES:]))

        # For the action b_a that has been done, 
        # select the value of q_eval, (q_eval originally has the value of all actions)
        q_eval = self.eval_net(b_s).gather(1, b_a)  # shape (batch, 1)

        q_next = self.target_net(b_s_).detach()  # q_next does not perform reverse transfer error, so detach
        q_target = b_r + self.GAMMA * q_next.max(1)[0].reshape(-1, 1)  # shape (batch, 1)
        loss = self.loss_func(q_eval, q_target)

        # Calculate, update eval net
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

# change every EPISODE times
EPISODE = 10
#throughput * 2, lost rate *2, len(status_episode[0]) * EPISODE
N_F = 2 + 2 + EPISODE * 6
# 1.4,1.1,0.4
N_A = 16
# random choose
Lambda_init = 0.9
# decline Lambda after random_counter
random_counter_init = 40

# standardlize to 1
MAX_BANDWITH = 30000
MIN_BANDWITH = 500

class RL(CongestionControl):

    def __init__(self):
        super(RL, self).__init__()
        self.USE_CWND=False
        self.send_rate = 500.0

        # list to store the input of "cc_trigger"
        self._input_list = []

        self.counter = 0 # EPISODE counter

        # block_dict[0]记录pac所属的block已经完成的包数，[1]记录丢包信息
        self.block_dict = {}

        self.blocks_priority = []

        self.result_list = []
        # state_episode记录每一个ack收集到的重要信息
        self.state_episode = []
        self.last_state = np.zeros(N_F)
        self.last_action = 1
        self.Lambda = Lambda_init
        self.random_counter = 0
        # self.random_counter = random_counter_init
        self.alpha = 1

        self.dqn = DQN(N_STATES=N_F,
                        N_ACTIONS=N_A,
                        LR=0.01,
                        GAMMA=0.9,
                        TARGET_REPLACE_ITER=100,
                        MEMORY_CAPACITY=500,
                        BATCH_SIZE=32
                    )
        
        # update in 2020-7-30
        self.event_nums = 0
        self.event_lost_nums = 0
        self.event_ack_nums = 0

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
        block_res_time = block_ddl + block_create_time - event_time
        block_res_nums = len(self.block_dict[event_block_id][1]) if event_block_id in self.block_dict else 0

        inflight = packet["Extra"]["inflight"]
        latency = packet["Latency"] + packet["Send_delay"] + packet["Pacing_delay"]
        packet_idx = packet["Offset"]


        if event_block_id in self.block_dict and packet_idx not in self.block_dict[event_block_id][1]:
            block_res_nums += block_split_nums - packet_idx

        pac_exp_rate = block_res_nums / block_res_time if block_res_time > 0 else 0

        if event_block_id in self.block_dict:
            self.block_dict[event_block_id][0] += 1
        else:
            self.block_dict[event_block_id] = [1, []]

        # Preparing for the bandwidth estimator's decision
        # self.latency_list.append(packet["Latency"] + packet["Send_delay"] + packet["Pacing_delay"])
        self.event_nums += 1

        if event_type == EVENT_TYPE_DROP:
            self.result_list.append(1)
            self.block_dict[event_block_id][1].append(packet_idx)
            self.event_lost_nums += 1
        else:
            self.result_list.append(0)
            self.event_ack_nums += 1

        info = []
        info.append(self.send_rate / MAX_BANDWITH)
        info.append(latency)
        info.append(inflight)
        info.append(self.cwnd)
        info.append(pac_exp_rate)
        info.append(len(self.block_dict[event_block_id][1]))
        self.state_episode.append(info)

        self.counter += 1
        if self.counter == EPISODE: # choose action every EPISODE times
            self.counter = 0
            # print()
            # print("EPISODE: send_rate is {}".format(self.send_rate))
            # print()

            # loss rate
            sum_loss_rate = self.event_lost_nums / self.event_nums

            # 10ms 内的丢包率
            instant_packet = list(filter(lambda item: self._input_list[-1][0] - item[0] < 0.01, self._input_list))
            instant_loss_nums = sum([1 for data in instant_packet if data[1]["event_type"] == 'D']) 
            instant_loss_rate = instant_loss_nums / len(instant_packet) if len(instant_packet) > 0 else 0

            # throughput
            sum_rate = self.event_ack_nums / event_time

            # 10ms 内的吞吐速率
            instant_ack_packet = list(filter(lambda data:data[1]["event_type"] == 'F', instant_packet))
            instant_rate = len(instant_ack_packet) / (instant_packet[-1][0] - instant_packet[0][0]) if (instant_packet[-1][0] - instant_packet[0][0]) > 0 else 0

            # declining random rate
            # self.random_counter -= 1
            # if self.random_counter <= 0:
            #     self.Lambda /= 2.0
            #     if self.Lambda < 0.05:
            #         self.Lambda = 0.05
            #     self.random_counter = 4

            self.random_counter += 1
            if self.random_counter > 100 and self.random_counter % 20 == 0:
                if self.Lambda >= 0.5:
                    self.Lambda *= 0.8
                elif 0.2 <= self.Lambda < 0.5:
                    self.Lambda *= 0.7
                elif 0.1 <= self.Lambda < 0.2:
                    self.Lambda *= 0.6
                else:
                    self.Lambda = self.Lambda * 0.5
            if self.Lambda < 0.05: self.Lambda = 0.05
            # print("self.random_counter: {}".format(self.random_counter))



            # current reward
            r = 0
            for i in range(EPISODE):
                if self.result_list[i] == 0:
                    if block_res_nums == 0 and block_res_time >= 0:
                        r += 1
                    r += self.send_rate / MAX_BANDWITH
                else:
                    r += - block_priority * self.alpha * block_res_time

            # current status
            s_ = []

            for i in range(EPISODE):
                for j in range(len(self.state_episode[0])):
                    s_.append(self.state_episode[i][j])

            s_.append(sum_loss_rate)
            s_.append(instant_loss_rate)
            s_.append(sum_rate)
            s_.append(instant_rate)

            s_array = np.array(s_)

            # store
            self.dqn.store_transition(self.last_state, self.last_action, r, s_array)

            # choose action
            a = self.dqn.choose_action(s_array)
           
            # exploration
            if random.random() < self.Lambda:
                a = random.randint(0, 15)

            if self.send_rate < MIN_BANDWITH:
                self.send_rate = MIN_BANDWITH
            if self.send_rate > MAX_BANDWITH:
                self.send_rate = MAX_BANDWITH

            # 速率变化： [0.5, 2] 倍区间
            self.send_rate *= (0.5 + 0.1 * a)



            self.last_action = a
            # DQN learn
            self.dqn.learn()

            self.last_state = s_
            self.result_list = []
            self.state_episode = []



# Your solution should include packet selection and bandwidth estimator.
# We recommend you to achieve it by inherit the objects we provided and overwritten necessary method.
class MySolution(BlockSelection, RL):

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
            packet_block_create_time = block.block_info["Create_time"]

            asume_time = self.rtt / 2
            asume_rate = self.send_rate

            best_block_priority = best_block.block_info["Priority"]
            block_priority = block.block_info["Priority"]

            best_block_offset = best_block.offset
            block_offset = block.offset

            best_block_remain_packets = best_block.block_info["Split_nums"] - best_block_offset + 1
            block_remain_packets = block.block_info["Split_nums"] - block_offset + 1

            is_retrans = block.retrans
            is_best_retrans = best_block.retrans

            if (block_remain_packets < 0):
                block_remain_packets = 0

            if best_block_remain_packets < 0:
                best_block_remain_packets = 0

            # if packet is miss ddl
            if is_retrans:
                if (cur_time - packet_block_create_time + asume_time + 1 / (asume_rate)) > block.block_info[
                    "Deadline"]:
                    return False
            else:
                if (cur_time - packet_block_create_time + asume_time + block_remain_packets / (asume_rate)) > \
                        block.block_info["Deadline"]:
                    return False
            if is_best_retrans:
                if (cur_time - best_block_create_time + asume_time + 1 / (asume_rate)) > best_block.block_info[
                    "Deadline"]:
                    return True
            else:
                if (cur_time - best_block_create_time + asume_time + best_block_remain_packets / (asume_rate)) > \
                        best_block.block_info["Deadline"]:
                    return True

            if (block.block_info["Block_id"] == best_block.block_info["Block_id"]):
                return best_block_offset < block_offset
            if best_block_priority != block_priority:
                return best_block_priority > block_priority
            else:
                if best_block_remain_packets != block_remain_packets:
                    return best_block_remain_packets > block_remain_packets

                if best_block_create_time != packet_block_create_time:
                    return best_block_create_time > packet_block_create_time
                else:
                    packet_ability = cur_time - packet_block_create_time + asume_time
                    packet_ability *= block.block_info["Deadline"]

                    best_packet_ability = cur_time - best_block_create_time + asume_time
                    best_packet_ability *= best_block.block_info["Deadline"]

                    return best_packet_ability > packet_ability

        best_block_idx = -1
        best_block = None
        for idx, item in enumerate(block_queue):
            if best_block is None or is_better(item):
                best_block_idx = idx
                best_block = item
        # print(self.cur_time, len(block_queue))
        # self.input_queue_length = len(block_queue)
        return best_block_idx

    def on_packet_sent(self, cur_time):
        """
        The part of solution to update the states of the algorithm, when sender need to send packet.
        """
        return super().on_packet_sent(cur_time)
    
    def cc_trigger(self, cur_time, event_info):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        super().estimate_bandwidth(cur_time, event_info)

        # set cwnd or sending rate in sender according to bandwidth estimator
        return {
            "cwnd" : self.cwnd,
            "send_rate" : self.send_rate,
        }

