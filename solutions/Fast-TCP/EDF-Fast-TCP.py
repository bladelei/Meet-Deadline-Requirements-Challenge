from simple_emulator import BlockSelection, cal_qoe

# We provided some simple algorithms about congestion control to help you being familiar with this competition.
# Like Reno and an example about reinforcement learning implemented by tensorflow
from simple_emulator import Reno
from simple_emulator import create_emulator
import random
import json
import math
import numpy as np
import os
import pathlib
import pandas as pd



class MySolution(BlockSelection, Reno):
    def __init__(self):
        super(MySolution, self).__init__()
        self.block_rest_size = dict()
        self.last_select_block_id = None
        # self.USE_CWND = False

        self.select_packet_call_num = 0
        self.make_decision_call_num = 0


        # FAST TCP hyperparameters
        self.new_cwnd_weight = 0.1
        self.linear_increase_size = 1.2

        # FAST TCP estimation parameters
        self.moving_average_weight = 0
        self.average_RTT = 0
        self.minimum_RTT = -1
        self.average_queuing_delay = 0

        self.cwnd = 6
        self.MIN_CWND = 2
        self.MAX_CWND = 5000



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
        best_block= None
        for idx, item in enumerate(block_queue):
            if best_block is None or is_better(item) :
                best_block_idx = idx
                best_block = item

        return best_block_idx

    def on_packet_sent(self, cur_time):
        """
        The part of solution to update the states of the algorithm when sender need to send packet.
        """
        return super().on_packet_sent(cur_time)

    def cc_trigger(self, cur_time, event_info):
        event_type = event_info["event_type"]
        event_time = cur_time
        packet = event_info["packet_information_dict"]

        RTT_sample = packet["Latency"] + packet["Pacing_delay"] + packet["Send_delay"]
        if self.minimum_RTT == -1:
            self.minimum_RTT = RTT_sample

        self.moving_average_weight = min(3/self.cwnd, 1/4)
        self.minimum_RTT = min(self.minimum_RTT, RTT_sample)

        self.average_queuing_delay = self.average_RTT - self.minimum_RTT

        self.average_RTT = ((1 - self.moving_average_weight) * self.average_RTT +
                            self.moving_average_weight * RTT_sample)

        self.cwnd = min(2 * self.cwnd,
                        ((1-self.new_cwnd_weight)*self.cwnd +
                         self.new_cwnd_weight*(1*self.minimum_RTT/self.average_RTT*self.cwnd +
                                               self.linear_increase_size)))

        if self.cwnd > self.MAX_CWND:
            self.cwnd = self.MAX_CWND
        if self.cwnd < self.MIN_CWND:
            self.cwnd = self.MIN_CWND

        # set cwnd or sending rate in sender
        return {
            "cwnd": self.cwnd,
            "send_rate": self.send_rate,
        }
