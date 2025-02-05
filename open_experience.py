"""
This demo aims to help player running system quickly by using the pypi library DTP-Emualtor https://pypi.org/project/DTP-Emulator/.
"""
import os, sys, inspect
import pathlib
import pandas as pd

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
sys.path.insert(0, currentdir)
from simple_emulator import SimpleEmulator, CongestionControl, create_2flow_emulator, create_emulator, analyze_emulator, \
    plot_cwnd, plot_rate
from simple_emulator import cal_qoe
# import os, sys, inspect
# currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
import datetime, pymysql, traceback, time, datetime
import numpy as np
# from random import seed
# import random
from func_timeout import func_set_timeout
import func_timeout

def get_network(idx):
    path = "/Users/bladexaver/AnsibleUI/DTP/Meet-Deadline-Requirements-Challenge/datasets/scenario_{}/networks/".format(idx)
    return [path + file for file in os.listdir(path)]

def get_dataset(block_mode=0, network_mode=0):
    pre_sub = '/Users/bladexaver/AnsibleUI/emulator/datasets/aitrans2/'
    all_blocks = [
        # ffmpeg 文件 general
        [pre_sub + "test/day_1/blocks/block-priority-0.csv",
         pre_sub + "test/day_1/blocks/block-priority-1.csv",
         pre_sub + "test/day_1/blocks/block-priority-2.csv"],

        # 音视频 webrtc
        [pre_sub + "test/day_2/blocks/block_audio.csv",
         pre_sub + "test/day_2/blocks/block_video.csv"],

        # 云游戏 game
        [pre_sub + "test/day_3/blocks/block-priority-0-ddl-0.15-.csv",
         pre_sub + "test/day_3/blocks/block-priority-1-ddl-0.5-.csv",
         pre_sub + "test/day_3/blocks/block-priority-2-ddl-0.2-.csv"]
    ]

    # create block trace
    blocks = all_blocks[block_mode]

    # create network trace
    networks = get_network(network_mode + 1)

    # pre = '/Users/bladexaver/AnsibleUI/emulator/datasets/mmgc/test/'
    background_traffics = [None,
                           "datasets/background_traffic_traces/live_pubg.csv",
                           "datasets/background_traffic_traces/movie_on_demand.csv",
                           "datasets/background_traffic_traces/web.csv"]

    # background_traffics = [None, pre + "background_traces/live_pubg.csv"]
    return {
        "blocks": blocks,
        "networks": networks,
        "background_traffics": background_traffics
    }


# @func_set_timeout(14400)
def evaluation(log_file, solution_file):
    res = []
    days = 3
    for idx in range(days):
        # idx=
        res_qoe = []
        datasets = get_dataset(idx, idx)
        # 格式化打印
        import pprint
        pp = pprint.PrettyPrinter(indent=2)
        print('datasets:')
        pp.pprint(datasets)

        blocks = datasets["blocks"]
        networks = datasets["networks"]
        background_traffics = datasets["background_traffics"]

        background = {}
        for v in background_traffics: background[v] = []

        '''
        # participants can try the task before competition, use following simple trace
        blocks = ["/home/team/competition/open_dataset/data_video.csv", "/home/team/competition/open_dataset/data_audio.csv"]
        networks = ["/home/team/competition/open_dataset/trace.txt"]
        '''
        solution = importlib.import_module(solution_file)

        # Use the object you created above
        my_solution = solution.MySolution()
        backgroud = {}
        for network in networks:
            num = network.split('/')[-1][7:-4]
            # num = 0
            for background_traffic in background_traffics:
                emulator = create_emulator(
                    block_file=blocks,
                    second_block_file=background_traffic,
                    trace_file=network,
                    solution=my_solution,
                    ENABLE_LOG=False,
                    SEED=int(num),
                    RUN_DIR=currentdir,
                    MIN_QUEUE=50,
                    MAX_QUEUE=500
                )
                emulator.run_for_dur(50)
                # print(currentdir)
                qoe = cal_qoe(run_dir=currentdir)
                print("scenario type: {}, network: {}, background_traffic: {}, qoe is {}".format(idx + 1, num, background_traffic, qoe))
                background[background_traffic].append(qoe)
                res_qoe.append(qoe)
        with open(log_file, "a+") as f:
            str_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            f.write(str_time)
            f.write("\nevaluation \nday:{0} qoe: {1}\n".format(idx, res_qoe))
            f.writelines("\nevaluation \nday:{0} background qoe: {1}\n".format(idx, background))

        print("*" * 40)
        print("background qoe is: {}".format(backgroud))
        print("*" * 40)
        res.append(np.mean(res_qoe))
    with open(log_file, "a+") as f:
        f.write("--------------------------------end--------------------------------\n\n\n")
    print('score of days(res):', res)
    # return res[0] / 1.5 + res[1] / 1 + res[2] / 2
    return res

def get_score(solution_file, type, log_name):
    # 设置测试数据集路径
    file_name_list = []
    score_list = []

    for test_day in [1, 2, 3]:
        BLOCK_BASE = 'datasets/scenario_' + str(test_day) + "/blocks/"
        NETWORK_BASE = 'datasets/scenario_' + str(test_day) + "/networks/"
        background_traffics = [None,
                               "datasets/background_traffic_traces/live_pubg.csv",
                               "datasets/background_traffic_traces/movie_on_demand.csv",
                               "datasets/background_traffic_traces/web.csv"]
        block_files = sorted([str(_) for _ in pathlib.Path(BLOCK_BASE).iterdir()])
        solution = importlib.import_module(solution_file)
        my_solution = solution.MySolution()
        for trace_name in sorted(os.listdir(NETWORK_BASE)):
            trace_file = pathlib.Path(NETWORK_BASE + trace_name)
            for background_traffic in background_traffics:
                emulator = create_emulator(
                    block_file=block_files,
                    second_block_file=background_traffic,
                    trace_file=trace_file,
                    solution=my_solution,
                    ENABLE_LOG=False,
                    SEED=int(trace_name[trace_name.find('_') + 1]),
                    RUN_DIR=currentdir,
                    MIN_QUEUE=50,
                    MAX_QUEUE=500
                )
                emulator.run_for_dur(50)
                # print(currentdir)
                qoe = cal_qoe(run_dir=currentdir)

                # file_name_list.append(str(test_day) + '_' + trace_name[trace_name.find('_') + 1:-4])
                score_list.append(qoe)

    # for file_name, score in zip(file_name_list, score_list):
    #     print(file_name, score)

    # with open('running_info_log/time_delay_cwnd.csv', 'wb') as f:
    pd.DataFrame(score_list).to_csv('running_log/{0}/{1}.csv'.format(type, log_name))
    # pd.DataFrame(score_list).to_csv('running_log/Hybrid/{}.csv'.format(log_name))


def get_scenaro_score(test_day, solution_file, log_name, loc):

    score_list = []

    BLOCK_BASE = 'datasets/scenario_' + str(test_day) + "/blocks/"
    NETWORK_BASE = 'datasets/scenario_' + str(test_day) + "/networks/"
    background_traffics = [None,
                           "datasets/background_traffic_traces/live_pubg.csv",
                           "datasets/background_traffic_traces/movie_on_demand.csv",
                           "datasets/background_traffic_traces/web.csv"]
    block_files = sorted([str(_) for _ in pathlib.Path(BLOCK_BASE).iterdir()])
    solution = importlib.import_module(solution_file)
    my_solution = solution.MySolution()
    for trace_name in sorted(os.listdir(NETWORK_BASE)):
        trace_file = pathlib.Path(NETWORK_BASE + trace_name)
        for background_traffic in background_traffics:
            emulator = create_emulator(
                block_file=block_files,
                second_block_file=background_traffic,
                trace_file=trace_file,
                solution=my_solution,
                ENABLE_LOG=False,
                SEED=int(trace_name[trace_name.find('_') + 1]),
                RUN_DIR=currentdir,
                MIN_QUEUE=50,
                MAX_QUEUE=500
            )
            emulator.run_for_dur(50)
            qoe = cal_qoe(run_dir=currentdir)

            score_list.append(qoe)

    pd.DataFrame(score_list).to_csv(f'running_log/{loc}/{log_name}.csv')

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



def get_scenarios(file_list, loc):

    for f in file_list:
        file_name = get_file_name(f)
        for day in range(1, 4):
            get_scenaro_score(day, f, file_name, f"{loc}_{day}")




if __name__ == "__main__":
    # sys.path.insert(0,currentdir)
    # import the solution
    import importlib

    #EDF
    EDF_NewReno = 'solutions.reno.EDF-NewReno'
    EDF_Fast_TCP = 'solutions.Fast-TCP.EDF-Fast-TCP'
    EDF_BBR = 'solutions.bbr.EDF-BBR'
    EDF_DRL_CC = 'solutions.dqn.EDF-DRL-CC'
    EDF_CUBIC = 'solutions.cubic.EDF-CUBIC'
    EDF_COPA = 'solutions.copa.EDF-COPA'

    # EDF_LIST = [EDF_NewReno, EDF_Fast_TCP, EDF_BBR, EDF_DRL_CC, EDF_CUBIC]
    EDF_LIST = [EDF_BBR]
    # get_scenarios(EDF_LIST, "E_scenario")
    # for e in EDF_LIST:
    #     get_score(e, 'EDF', get_file_name(e))

    # file_name = get_file_name(EDF_BBR)
    # get_scenaro_score(3, EDF_BBR, file_name, "E_scenario_3")



    # Hybrid
    EDF_NewReno = 'solutions.reno.EDF-NewReno'
    DTP_Fast_TCP = 'solutions.Fast-TCP.DTP-Fast-TCP'
    HPF_BBR = 'solutions.bbr.HPF-BBR'
    DRL_TC = 'solutions.dqn.DRL-TC'

    H_LIST = [EDF_NewReno, DTP_Fast_TCP, HPF_BBR, DRL_TC]

    HPF_DRL_CC = 'solutions.dqn.HPF-DRL-CC'
    DTP_DRL_CC = 'solutions.dqn.DTP-DRL-CC'


    #
    # get_scenarios(S_LIST, "S_scenario")
    # get_scenarios(H_LIST, "H_scenario")
    # for h in H_LIST:
    #     get_score(h, 'Hybrid', get_file_name(h))
    S_LIST = [HPF_DRL_CC, DTP_DRL_CC]
    get_scenarios(S_LIST, "S_scenario")
    for s in S_LIST:
        get_score(s, 'Select', get_file_name(s))



    # evaluation
    # log_file = "./prelimelary.log"
    #
    # evaluation_begin = time.time()
    #
    # res = evaluation(log_file, solution_file)
    #
    # evaluation_end = time.time()
    # print('time used: ', evaluation_end - evaluation_begin, ' s')
    # print("最终QoE结果：{}".format(res))


    # get_score(DTP_Fast_TCP, "DTP_Fast_TCP")
    # get_score(EDF_CUBIC, "EDF_CUBIC")
    # get_score(EDF_COPA, "EDF_COPA")

