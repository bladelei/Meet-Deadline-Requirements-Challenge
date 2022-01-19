"""
This demo aims to help player running system quickly by using the pypi library simple-emualtor https://pypi.org/project/simple-emulator/.
"""
from simple_emulator import CongestionControl

# We provided a simple algorithms about block selection to help you being familiar with this competition.
# In this example, it will select the block according to block's created time first and radio of rest life time to deadline secondly.
from simple_emulator import BlockSelection

# We provided some simple algorithms about congestion control to help you being familiar with this competition.
# Like Reno and an example about reinforcement learning implemented by tensorflow
from simple_emulator import Reno

EVENT_TYPE_FINISHED = 'F'
EVENT_TYPE_DROP = 'D'
EVENT_TYPE_TEMP = 'T'


# Your solution should include block selection and bandwidth estimator.
# We recommend you to achieve it by inherit the objects we provided and overwritten necessary method.
class MySolution(BlockSelection, Reno):

    def __init__(self):
        super().__init__()
        # base parameters in CongestionControl

        # the value of congestion window
        self.cwnd = 10


        # use cwnd
        self.USE_CWND = True

        # current time
        self.cur_time = -1
        # the value of cwnd at last packet event
        self.last_cwnd = 0
        # the number of lost packets received at the current moment
        self.instant_drop_nums = 0

        # the number of lost packets
        self.drop_nums = 0
        # the number of acknowledgement packets
        self.ack_nums = 0

        # use in select_block
        self.rtt = 0

        self.call_nums = 0

        self.maxbw = float("-inf")
        self.minrtt = float("inf")

        self.bbr_mode = ["BBR_STARTUP", "BBR_DRAIN", "BBR_PROBE_BW", "BBR_PROBE_RTT"]
        self.mode = "BBR_STARTUP"

        # Window length of bw filter (in rounds)
        self.bbr_bw_rtts = 10

        # Window length of min_rtt filter (in sec)
        self.bbr_min_rtt_win_sec = 5

        # Minimum time (in s) spent at bbr_cwnd_min_target in BBR_PROBE_RTT mode
        self.bbr_probe_rtt_mode_s = 0.2

        self.bbr_high_gain = 2885 / 1000 + 1
        self.bbr_drain_gain = 1000 / 2885

        # probe_bw
        self.bbr_cwnd_gain = 2
        self.probe_bw_gain = [5 / 4, 4 / 3, 1, 1, 1, 1, 1, 1]
        self.cycle_index = 0

        self.probe_rtt_gain = 1

        self.bbr_min_cwnd = 4

        self.pacing_gain = self.bbr_high_gain
        self.cwnd_gain = self.bbr_high_gain

        # to check when the mode come to drain
        self.four_bws = [0] * 4

        # the start time of probe rtt
        self.probe_rtt_time = 0
        self.delivered_nums = 0

        # used to check when come to PROBE_RTT mode
        self.ten_sec_wnd = []

        #  high_gain * init_cwnd / RTT
        self.pacing_rate = self.bbr_high_gain * self.cwnd / 0.002
        # for sampling
        self.bw_windows = []

        # the value of sending rate
        self.send_rate = self.bbr_high_gain * self.cwnd / 0.002


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
        # super().on_packet_sent(cur_time)
        self.call_nums += 1
        output = {
            "cwnd": self.cwnd,
            "send_rate": self.pacing_rate,
            "pacing_rate": float("inf"),
            "extra": {
                "delivered": self.delivered_nums,
                # "pacing_rate": self.pacing_rate,
                "pacing_gain": self.pacing_gain,
                "cwnd_gain": self.cwnd_gain,
                "max_bw": self.maxbw,
                "min_rtt": self.minrtt,
                "mode": self.mode,
                "cycle_index": self.cycle_index
            }
        }
        return output

    def get_max_bw(self):
        return max(self.bw_windows)

    def append_bw(self, now_bw):
        self.bw_windows.append(now_bw)
        # keep the latest 10 bw
        if len(self.bw_windows) > self.bbr_bw_rtts:
            self.bw_windows.pop(0)

    # calculate rtt and bw on ack
    def cal_bw(self, send_delivered, rtt):
        delivered = self.delivered_nums - send_delivered
        return delivered / rtt

    def stop_increasing(self, bws):
        if len(bws) < 4:
            return False
        thresh = 0.1
        scale1 = (bws[1] - bws[0]) / bws[0]
        scale2 = (bws[2] - bws[1]) / bws[1]
        scale3 = (bws[3] - bws[2]) / bws[2]
        return scale1 < thresh and scale2 < thresh and scale3 < thresh

    def update_min_rtt(self, event_time):
        # making sure the rtt data is in 10s
        # the last item is from now packet, so their difference=0
        while event_time - self.ten_sec_wnd[0][0] >= self.bbr_min_rtt_win_sec:
            self.ten_sec_wnd.pop(0)
        idx = -1
        for i, time_rtt in enumerate(self.ten_sec_wnd):
            if idx == -1 or time_rtt[1] <= self.ten_sec_wnd[idx][1]:
                idx = i
        # now rtt is not the minist
        if idx == -1 or idx != len(self.ten_sec_wnd) - 1:
            return False
        # update min rtt in new round
        self.minrtt = self.ten_sec_wnd[idx][1]
        # begin with the time with min rtt
        self.ten_sec_wnd = self.ten_sec_wnd[idx:]
        return True

    def set_output(self, mode):

        # it seems that there is a minest pacing rate
        # ref : https://code.woboq.org/linux/linux/net/ipv4/tcp_bbr.c.html#259

        self.pacing_rate = self.pacing_gain * self.maxbw
        self.cwnd = max(self.maxbw * self.minrtt * self.cwnd_gain, 4)

    def cal_gain(self, mode):
        if mode == self.bbr_mode[0]:
            pacing_gain = self.bbr_high_gain
            cwnd_gain = self.bbr_high_gain

        elif mode == self.bbr_mode[1]:
            pacing_gain = self.bbr_drain_gain
            cwnd_gain = self.bbr_high_gain

        elif mode == self.bbr_mode[2]:
            pacing_gain = 1 if self.stop_increasing(self.four_bws) else self.probe_bw_gain[self.cycle_index]
            cwnd_gain = self.bbr_cwnd_gain
            self.cycle_index += 1
            if self.cycle_index == len(self.probe_bw_gain):
                self.cycle_index = 1

        elif mode == self.bbr_mode[3]:
            pacing_gain = 1
            cwnd_gain = 1
        return pacing_gain, cwnd_gain

    def cc_trigger(self, cur_time, event_info):
        """
        The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
        See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
        """

        event_type = event_info["event_type"]
        event_time = cur_time
        packet = event_info["packet_information_dict"]
        packet_rtt = packet["Latency"] + packet["Send_delay"] + packet["Pacing_delay"]
        rtt = packet["Latency"] + packet["Pacing_delay"] + packet["Send_delay"]
        alf = 0.9

        if self.cur_time < event_time:
            # initial parameters at a new moment
            self.last_cwnd = 0
            self.instant_drop_nums = 0

        # if packet is dropped
        # if event_type == EVENT_TYPE_DROP:
        #     # dropping more than one packet at a same time is considered one event of packet loss
        #     if self.instant_drop_nums > 0:
        #         return
        #     self.instant_drop_nums += 1
        #     self.drop_nums += 1
        #     # clear acknowledgement count
        #     self.ack_nums = 0
        #     # Ref 1 : For ensuring the event type, drop or ack?
        #     self.cur_time = event_time
        #     if self.last_cwnd > 0 and self.last_cwnd != self.cwnd:
        #         # rollback to the old value of cwnd caused by acknowledgment first
        #         self.cwnd = self.last_cwnd
        #         self.last_cwnd = 0

        # if packet is acknowledged
        if event_type == EVENT_TYPE_FINISHED or event_type == EVENT_TYPE_DROP:
            # Ref 1
            if event_time <= self.cur_time:
                return
            self.cur_time = event_time
            self.last_cwnd = self.cwnd

            # update rtt
            # self.rtt = alf * packet_rtt + (1 - alf) * self.rtt

            self.delivered_nums += 1

            send_delivered = packet["Extra"]["delivered"]
            # update bandwidth
            bw = self.cal_bw(send_delivered, rtt)
            # if is the first ack
            if self.maxbw == float("-inf"):
                self.maxbw = bw
                self.minrtt = rtt
            self.append_bw(bw)
            self.four_bws = self.bw_windows[-4:]
            self.maxbw = self.get_max_bw()

            if self.mode == self.bbr_mode[0]:
                if self.stop_increasing(self.four_bws):
                    self.mode = self.bbr_mode[1]

            if self.mode == self.bbr_mode[1]:
                inflight = packet["Extra"]["inflight"]
                BDP = self.maxbw * self.minrtt
                if BDP >= inflight:
                    self.mode = self.bbr_mode[2]
                    self.cycle_index = 0

            if self.mode == self.bbr_mode[3]:
                if event_time - self.probe_rtt_time >= self.bbr_probe_rtt_mode_s:
                    if self.stop_increasing(self.four_bws):
                        self.mode = self.bbr_mode[2]
                        self.cycle_index = 1
                    else:
                        self.mode = self.bbr_mode[0]
            # update RTT
            # value of ten_sec_wnd
            time_rtt = [event_time, rtt]
            self.ten_sec_wnd.append(time_rtt)
            # rtt window exceed
            if event_time - self.ten_sec_wnd[0][0] >= self.bbr_min_rtt_win_sec:
                flag = self.update_min_rtt(event_time)
                # now rtt is not the minest, so enter prob_rtt
                if (not flag) and self.mode != self.bbr_mode[3]:
                    self.mode = self.bbr_mode[3]
                    self.cwnd = self.bbr_min_cwnd
                    self.probe_rtt_time = event_time

            # find new min rtt in bbr_min_rtt_win_sec
            elif rtt < self.ten_sec_wnd[0][1]:
                self.minrtt = rtt
                self.ten_sec_wnd = self.ten_sec_wnd[-1:]
            # update gains
            self.pacing_gain, self.cwnd_gain = self.cal_gain(self.mode)
            self.set_output(self.mode)
            # self.rtt = alf * self.rtt + (1 - alf) * self.minrtt

        # set cwnd or sending rate in sender
        return {
            "cwnd": self.cwnd,
            "send_rate": self.pacing_rate,
        }
