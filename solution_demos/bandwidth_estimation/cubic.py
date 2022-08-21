"""
This demo aims to help player running system quickly by using the pypi library DTP-Emualtor https://pypi.org/project/DTP-Emulator/.
"""
from simple_emulator import CongestionControl

# We provided a simple algorithms about packet selection to help you being familiar with this competition.
# In this example, it will select the packet according to packet's created time first and radio of rest life time to deadline secondly.

# We provided some simple algorithms about congestion control to help you being familiar with this competition.
# Like Reno and an example about reinforcement learning implemented by tensorflow
from simple_emulator import Reno

import time
import math

EVENT_TYPE_FINISHED = 'F'
EVENT_TYPE_DROP = 'D'
EVENT_TYPE_TEMP = 'T'

# cubic
TCP_FRIENDLINESS = 0
Fast_CONVERGENCE = 0
BETA = 0.2
CUBIC = 0.4


# Your solution should include packet selection and congestion control.
# So, we recommend you to achieve it by inherit the objects we provided and overwritten necessary method.
class Cubic(Reno):

    def __init__(self):
        # base parameters in CongestionControl
        self._input_list = []
        self.cwnd = 1
        self.send_rate = float("inf")
        self.pacing_rate = float("inf")
        self.call_nums = 0

        # for reno
        self.ssthresh = 10
        self.curr_state = "slow_start"
        self.states = ["slow_start", "congestion_avoidance", "fast_recovery"]
        self.drop_nums = 0
        self.ack_nums = 0

        self.cur_time = -1
        self.last_cwnd = 0
        self.instant_drop_nums = 0

        # cubic
        self.last_max_cwmd = 65
        self.epoch_start = 0
        self.origin_point = 0
        self.d_min = 0
        self.tcp_cwmd = 0
        self.K = 0.0
        self.ack_cnt = 0
        self.cwnd_cnt = 0
        self.last_time = 0
        self.cnt = 0.0

    def make_decision(self, cur_time):
        """
        The part of algorithm to make congestion control, which will be call when sender need to send pacekt.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        return super().make_decision(cur_time)

    def cc_trigger(self, data):

        # packet loss
        def cubic_rest():
            # self.last_max_cwmd = 0
            self.epoch_start = 0
            self.origin_point = 0
            self.d_min = 0
            self.tcp_cwmd = 0
            self.K = 0.0
            self.ack_cnt = 0.0
            self.cwnd_cnt = 0
            self.last_time = 0
            self.cnt = 0.0

        # calculate cubic root
        def cubic_root(num):
            return pow(num, 1.0 / 3)

        # Compute congestion window to use.
        def cubic_update():
            # count the total number of acks
            self.ack_cnt += 1
            # the current window is the same as the history window AND Time difference less than 1000 / 32ms
            if self.cwnd == self.last_cwnd and float(time.time()) - self.last_time < 1000 / (32.0*1000):
                return
            # record the window value when entering congestion avoidance
            self.last_cwnd = self.cwnd
            # record the time when entering congestion avoidance
            self.last_time = float(time.time())
            # enter congestion avoidance phase
            if self.epoch_start <= 0:
                # record current time
                self.epoch_start = float(time.time())
                self.ack_cnt = 1
                self.tcp_cwmd = self.cwnd
                if self.cwnd < self.last_max_cwmd:
                    self.K = cubic_root((self.last_max_cwmd - self.cwnd) / CUBIC)
                    self.origin_point = self.last_max_cwmd
                else:
                    self.K = 0.0
                    self.origin_point = self.cwnd

            current_time = float(time.time())
            t = current_time + self.d_min - self.epoch_start

            # |t-K|
            if t < self.K:
                offs = self.K - t
            else:
                offs = t - self.K
            # c*|t-k|^3
            delta = CUBIC * offs * offs * offs
            if t < self.K:
                target = self.origin_point - delta
            else:
                target = self.origin_point + delta

            if target > self.cwnd:
                self.cnt = self.cwnd / (target - self.cwnd)
            else:
                self.cnt = 100 * self.cwnd

            # tcp_friendliness
            if TCP_FRIENDLINESS:
                self.tcp_cwmd = self.tcp_cwmd + ((3 * BETA) / (2 - BETA)) * (self.ack_cnt / self.cwnd)
                self.ack_cnt = 0
                if self.cwnd < self.tcp_cwmd:
                    max_cnt = self.cwnd / (self.tcp_cwmd - self.cwnd)
                    if self.cnt > max_cnt:
                        self.cnt = max_cnt
            if self.cnt == 0.0:
                self.cnt = 1.0

        event_type = data["event_type"]
        event_time = data["event_time"]

        packet_info = data["packet_information_dict"]

        # on each ACK
        if event_type == EVENT_TYPE_FINISHED:
            # find min rtt
            min_rtt = packet_info["Latency"]
            if self.d_min:
                self.d_min = min(self.d_min, min_rtt)
            else:
                self.d_min = min_rtt

            if self.cwnd <= self.ssthresh:
                self.cwnd += 1
            else:
                print(self.cwnd)
                cubic_update()
                if self.cwnd_cnt > self.cnt:
                    self.cwnd += 1
                    self.cwnd_cnt = 0
                else:
                    self.cwnd_cnt += 1

        if event_type == EVENT_TYPE_DROP:
            cubic_rest()
            print('#', self.cwnd)
            # print(self.last_max_cwmd)
            if self.cwnd < self.last_max_cwmd and Fast_CONVERGENCE:
                self.last_max_cwmd = (self.cwnd * (2 - BETA) / 2)
            else:
                self.last_max_cwmd = self.cwnd
            self.cwnd = max((self.cwnd * (1 - BETA)), 1)
            self.ssthresh = max(self.cwnd, 2)

    def append_input(self, data):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """
        self._input_list.append(data)

        if data["event_type"] != EVENT_TYPE_TEMP:
            self.cc_trigger(data)
            return {
                "cwnd": self.cwnd,
                "send_rate": self.send_rate
            }
        return None

