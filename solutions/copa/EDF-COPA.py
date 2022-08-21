"""
This demo aims to help player running system quickly by using the pypi library DTP-Emualtor https://pypi.org/project/DTP-Emulator/.
"""
# from simple_emulator import PccEmulator, CongestionControl

# We provided a simple algorithms about packet selection to help you being familiar with this competition.
# In this example, it will select the packet according to packet's created time first and radio of rest life time to deadline secondly.
from simple_emulator import BlockSelection

# We provided some simple algorithms about congestion control to help you being familiar with this competition.
# Like Reno and an example about reinforcement learning implemented by tensorflow
from simple_emulator import Reno
# Ensuring that you have installed tensorflow before you use it
# from simple_emulator import RL

# We provided some function of plotting to make you analyze result easily in utils.py
# from simple_emulator import analyze_pcc_emulator, plot_rate
from simple_emulator import constant
import sys
from simple_emulator import cal_qoe
from collections import deque

import matplotlib.pyplot as plt

EVENT_TYPE_FINISHED='F'
EVENT_TYPE_DROP='D'
EVENT_TYPE_TEMP='T'
DIRECTION_TYPE_DOWN = 0 ### DOWN
DIRECTION_TYPE_UP = 1   ### UP
MAX_TIME = 500
# Your solution should include packet selection and congestion control.
# So, we recommend you to achieve it by inherit the objects we provided and overwritten necessary method.

class ExtremeWindow(object):
    def __init__(self,find_min):
        ### whether to find the maximum or minimum
        self.find_min = find_min
        ### maximum time till which to maintain window
        self.max_time = MAX_TIME
        self.vals = deque()  #用来存储rtt
        self.extreme = sys.maxsize if find_min==True else -sys.maxsize


    def clear_old_hist(self,now):
        recompute = False
        # Delete all samples older than max_time. However, if there is only one
        # sample left, don't delete it
        while(len(self.vals)>1 and self.vals[0][0]< now-self.max_time):
            if((self.find_min and self.vals[0][1]<=self.extreme) or \
                (not self.find_min and self.vals[0][1] >= self.extreme)):
                recompute = True
            self.vals.popleft()
        if recompute :
            self.extreme = self.vals[0][1]

    def new_sample(self,val,now):
        #Delete any RTT samples immediately before this one that are greater (or
        #less than if finding max) than the current value
        while(self.vals and ((self.find_min and self.vals[-1][1] > val) or \
            (not self.find_min and self.vals[-1][1] < val))):
            self.vals.pop()
        ## push back current sample and update extreme
        self.vals.append((now,val))
        if ((self.find_min and val < self.extreme) or (not self.find_min and val > self.extreme)):
            self.extreme = val
        ### delete unnecessary history
        self.clear_old_hist(now)


    def update_max_time(self,t):
        self.max_time = t

    def clear(self):
        self.max_time = MAX_TIME
        self.vals.clear()
        self.extreme = sys.maxsize if self.find_min==True else -sys.maxsize



class RTTWindow(object):
    def __init__(self):
        self.srtt=0
        self.srtt_alpha=1/16
        self.latest_rtt=0
        self.min_rtt=ExtremeWindow(True)
        self.unjitter_rtt = ExtremeWindow(True)
        self.is_copa_min=ExtremeWindow(True)
        self.is_copa_max=ExtremeWindow(False)

    def new_rtt_sample(self,rtt,now):
        ### update smoothed RTT
        if(self.srtt == 0):
            self.srtt = rtt
        self.srtt = self.srtt_alpha * rtt + (1- self.srtt_alpha)*self.srtt
        self.latest_rtt = rtt
        ### Update extreme value trackers
        max_time = max(MAX_TIME,20*self.srtt)
        self.min_rtt.update_max_time(max_time)
        self.unjitter_rtt.update_max_time(min(max_time,self.srtt*1))
        self.is_copa_min.update_max_time(min(max_time,self.srtt*4))
        self.is_copa_max.update_max_time(min(max_time,self.srtt*4))

        self.min_rtt.new_sample(rtt,now)
        self.unjitter_rtt.new_sample(rtt,now)
        self.is_copa_min.new_sample(rtt,now)
        self.is_copa_max.new_sample(rtt,now)

    def get_min_rtt(self):
        return self.min_rtt.extreme
    def get_unjitter_rtt(self):
        return self.unjitter_rtt.extreme
    def get_latest_rtt(self):
        return self.latest_rtt
    def is_copa(self):
        threshhold = self.min_rtt + 0.1*(self.is_copa_max - self.min_rtt)
        return self.is_copa_min.extreme < threshhold

    def clear(self):
        self.srtt = 0
        self.min_rtt.clear()
        self.unjitter_rtt.clear()
        self.is_copa_min.clear()
        self.is_copa_max.clear()

class ReduceOnLoss(object):
    def __init__(self):
        self.num_lost=0
        self.num_pkt=0
        self.prev_win_time=0
    def update(self, loss, cur_time, rtt):
        if(loss):
            self.num_lost +=1
        self.num_pkt += 1
        if(cur_time > self.prev_win_time + 2*rtt and self.num_pkt>20):
            loss_Rate = 1 * self.num_lost / self.num_pkt
            self.prev_win_time = cur_time
            self.num_lost = 0
            self.num_pkt = 0
            if(loss_Rate>0.3):
                return True
        return  False

    def reset(self):
        self.num_lost = 0
        self.num_pkt = 0
        self.prev_win_time = 0

class IsUniformDistr(object):
    def __init__(self,window_len):
        self.window_len=window_len
        self.window = deque()
        self.sum=0

    def combinatoral_nck(self, n, k):
        res = 1
        if k > n/2:
            k = n-k

        for i in range(n-k+1,n+1):
            res *= i
        for ki in range(1, k+1):
            res /= i
        return  res

    def update(self, data):
        self.window.append(data)
        self.sum+=data
        if(len(self.window) > self.window_len):
            self.sum -= self.window[0]
            self.window.popleft()

    def reset(self):
        self.window.clear()
        self.sum=0

class MySolution(BlockSelection, Reno):

    def __init__(self):
        super().__init__()
        self.copa_init()
        # base parameters in CongestionControl
        # the data appended in function "append_input"
        self._input_list = []
        # the value of crowded window
        self.cwnd = self.num_probe_pkts
        # the value of sending rate
        self.send_rate = float("inf")
        # the value of pacing rate
        self.pacing_rate = float("inf")
        # use cwnd
        self.USE_CWND=True

        # for reno
        self.ssthresh = float("inf")
        self.curr_state = "slow_start"
        self.states = ["slow_start", "congestion_avoidance", "fast_recovery"]
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

        ###画图
        self.rtt_record = []
        self.queuing_delay_record = []
        self.min_rtt_record = []
        self.unjitter_rtt_record = []

        self.queuing_delay = 0

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


    def copa_init(self):
        '''
        alogrithm with copa congestion conrol
        extra value:
        '''
        ### some adjustable parameters
        self.alpha_rtt = 1/1
        self.alpha_loss = 1/2
        self.alpha_rtt_long_avg = 1/4
        self.rtt_averaging_interval = 0.1
        self.num_probe_pkts = 10
        self.copa_k = 2


        ### dynamic value
        ### bool variable using to decide when to slow start .
        self.slow_start = True
        self.do_slow_start = True


        ### using to update cwnd (update intersend time )
        self.update_amt = 1 ## amt is v in copa paper
        self.prev_update_dir = 1
        self.update_dir = 1
        self.pkts_per_rtt = 0
        self.last_update_time = 0
        self.cur_intersend_time = 0
        self.intersend_time = 0
        self.rtt_window = RTTWindow()


        ## using to update delta function
        ## using python dict to alter cpp emu
        self.utility_mode_list = ["CONSTANT_DELTA", "PFABRIC_FCT", "DEL_FCT", "BOUNDED_DELAY", "BOUNDED_DELAY_END",
                            "MAX_THROUGHPUT", "BOUNDED_QDELAY_END", "BOUNDED_FDELAY_END", "TCP_COOP",
                            "CONST_BEHAVIOR", "AUTO_MODE"]

        self.operation_mode_list = ["DEFAULT_MODE", "LOSS_SENSITIVE_MODE", "TCP_MODE"]
        self.utility_mode=self.utility_mode_list[0]
        self.operation_mode = self.operation_mode_list[0]
        self.delta=0.5  ### using to define the importance between throughput and delay, it will grows up with the importance of throughput
        self.default_delta=0.5
        self.prev_delta_update_time=0.0
        self.prev_delta_update_time_loss=0.0
        self.is_uniform = IsUniformDistr(32)

        self.min_rtt = sys.maxsize
        # self.RTTstanding = BIGNUM
        # self.RTTmin = BIGNUM

        if (self.utility_mode !=self.utility_mode_list[1] ): #hui
            self.delta = 1

        #### ONACK
        self.reduce_on_loss = ReduceOnLoss()


    def update_delta(self,pkt_lost,cur_rtt):
        #pkt_lost --- bool  cur_rtt  --- double
        if self.utility_mode == self.utility_mode_list[10] :
            if pkt_lost:
                self.is_uniform.upadate(self.rtt_window.get_unjitter_rtt())
            if not self.rtt_window.is_copa():
                self.operation_mode == self.operation_mode_list[1]
            else:
                self.operation_mode == self.operation_mode_list[0]

        if self.operation_mode == self.operation_mode_list[0] and self.utility_mode == "TCP_COOP":
            if self.prev_delta_update_time == 0 or self.prev_delta_update_time_loss +cur_rtt < self.cur_time:
                if self.delta < self.default_delta:
                    self.delta = self.default_delta
                self.delta = min(self.delta, self.default_delta)
                self.prev_delta_update_time = self.cur_time
        elif self.utility_mode == "TCP_COOP" or self.operation_mode == "LOSS_SENSITIVE_MODE":
            if self.prev_delta_update_time == 0 :
                self.delta = self.default_delta
            if pkt_lost and self.prev_delta_update_time_loss +cur_rtt < self.cur_time:
                self.delta *=2
                self.prev_delta_update_time_loss = self.cur_time
            else:
                if self.prev_delta_update_time + cur_rtt < self.cur_time:
                    self.delta = 1 / (1 / self.delta+1)
                    self.prev_delta_update_time = self.cur_time
            self.delta = min(self.delta, self.default_delta)
        ### for test use simple delta
        # self.delta = 0.7


    def update_intersend_time(self):
        if(self.ack_nums < 2 * self.num_probe_pkts - 1):
            return
        rtt = self.rtt_window.get_unjitter_rtt()
        #("rtt",rtt)
        queuing_delay = rtt - self.min_rtt  #估计队列延迟
        self.queuing_delay = queuing_delay

        target_window = 0
        if (queuing_delay == 0):
            target_window = sys.maxsize
        else:
            target_window = rtt / (queuing_delay*self.delta)

        #print("queuing_delay:",queuing_delay,"target_window:",target_window)
        self.unjitter_rtt_record.append(rtt)
        self.queuing_delay_record.append(queuing_delay)
        self.min_rtt_record.append(self.min_rtt)

        if(self.slow_start):
            self.prev_ack_time = 0
            if (self.do_slow_start or target_window == sys.maxsize):
                self.cwnd *=2
                if(self.cwnd > target_window or self.rtt_window.get_unjitter_rtt() > self.rtt_window.get_min_rtt()):
                    self.slow_start = False
                    self.cwnd = (self.cwnd+target_window)/2
            else:
                assert(False)

            #print("slow_start:",self.cwnd)
        else:
            ### using to update v in copa code  amt == v
            if(self.last_update_time + self.rtt_window.get_latest_rtt() < self.cur_time):
                ## update v if it has not been update over an RTT
                if (self.prev_update_dir * self.update_dir > 0):
                    if self.update_amt < 0.006:
                        self.update_amt += 0.005
                    else:
                        self.update_amt = int(self.update_amt)
                else:
                    self.update_amt = 1
                    self.prev_update_dir *= -1
                self.last_update_time = self.cur_time
                self.pkts_per_rtt = self.update_dir = 0

            if self.update_amt > (self.cwnd * self.delta):
                self.update_amt /= 2
                self.update_amt = max(self.update_amt,1)
                self.pkts_per_rtt +=1

            # update cwnd and decide the update direction
            if self.cwnd < target_window:
                self.update_dir+=1
                self.cwnd += self.update_amt / (self.delta * self.cwnd)
                #print("increase cwnd",self.cwnd)
            else:
                self.update_dir-=1
                self.cwnd -= self.update_amt / (self.delta* self.cwnd)
                #print("decrease cwnd", self.cwnd)

            self.cur_intersend_time = 0.5* rtt/ self.cwnd

            ### update sending rate
            self.send_rate = 2*self.cwnd/rtt


    def cc_trigger(self, cur_time, event_info):

        event_type = event_info["event_type"]
        event_time = cur_time
        self.cur_time = event_time

        rtt = event_info["packet_information_dict"]["Latency"]
        self.rtt_record.append(rtt)

        self.rtt_window.new_rtt_sample(rtt, self.cur_time)
        self.min_rtt = self.rtt_window.get_min_rtt()
        #print("min_rtt：",self.min_rtt,"rtt:",rtt)
        self.prev_ack_time = self.cur_time
        self.update_delta(False, self.rtt_window.get_latest_rtt())
        self.update_intersend_time()

        pkt_lost = False
        reduce_window = False

        if event_type == EVENT_TYPE_DROP:
            self.update_delta(True, self.rtt_window.get_latest_rtt())
            reduce_window |= self.reduce_on_loss.update(True, self.cur_time, self.rtt_window.get_unjitter_rtt())
        reduce_window |= self.reduce_on_loss.update(False, self.cur_time, self.rtt_window.get_unjitter_rtt())

        if (reduce_window):
            self.cwnd *= 0.7
            self.cwnd = max(2, self.cwnd)
            self.cur_intersend_time = 0.5 * self.rtt_window.get_unjitter_rtt() / self.cwnd
            self.send_rate = 2 * self.cwnd / self.rtt_window.get_unjitter_rtt()
            #print("reduce cwnd", self.cwnd)
        self.ack_nums +=1

        return {
            "cwnd": self.cwnd,
            "send_rate": self.send_rate
        }



    # def append_input(self, data):   ### work for congestion control for by using cc trigger
    #     """
    #     The part of algorithm to make congestion control, which will be call when sender get an event about acknowledge or lost from reciever.
    #     See more at https://github.com/AItransCompetition/simple_emulator/tree/master#congestion_control_algorithmpy.
    #     """
    #     # add new data to history data
    #     self._input_list.append(data)
    #     if data["event_type"] != EVENT_TYPE_TEMP:
    #         # specify congestion control algorithm
    #         #self.cc_trigger(data)
    #         self.cc_trigger(data)
    #         #print("cwnd",self.cwnd)
    #         #print("send_rate", self.send_rate)
    #         # set cwnd or sending rate in sender
    #         self.cwnd = sys.maxsize
    #         self.send_rate = sys.maxsize
    #         return {
    #             "cwnd" : self.cwnd,
    #             "send_rate" : self.send_rate
    #         }
    #     return None



# if __name__ == '__main__':
#     # fixed random seed
#     import random
#     random.seed(1)
#
#     # The file path of packets' log
#     log_packet_file = "output/packet_log/packet-0.log"
#
#     # Use the object you created above
#     my_solution = MySolution()
#
#     # Create the emulator using your solution
#     # Specify USE_CWND to decide whether or not use crowded windows. USE_CWND=True by default.
#     # Specify ENABLE_LOG to decide whether or not output the log of packets. ENABLE_LOG=True by default.
#     # You can get more information about parameters at https://github.com/AItransCompetition/simple_emulator/tree/master#constant
#     emulator = PccEmulator(
#         block_file=["traces/block-priority-0.csv","traces/block-priority-1.csv","traces/block-priority-2.csv"],
#         trace_file="traces/traces_3.txt",
#         solution=my_solution,
#         # enable logging packet. You can train faster if USE_CWND=False
#         ENABLE_LOG=True
#     )
#
#     # Run the emulator and you can specify the time for the emualtor's running.
#     # It will run until there is no packet can sent by default.
#     emulator.run_for_dur(20)
#
#     # print the debug information of links and senders
#     emulator.print_debug()
#
#     # Output the picture of emulator-analysis.png
#     # You can get more information from https://github.com/AItransCompetition/simple_emulator/tree/master#emulator-analysispng.
#     analyze_pcc_emulator(log_packet_file, file_range="all")
#
#     # Output the picture of rate_changing.png
#     # You can get more information from https://github.com/AItransCompetition/simple_emulator/tree/master#cwnd_changingpng
#     plot_rate(log_packet_file, trace_file="traces/traces_3.txt", file_range="all", sender=[1])
#
#
#     print("Qoe : %d" % (cal_qoe()) )
