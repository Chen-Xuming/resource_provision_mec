import simplejson as json
from matplotlib import pyplot as plt
import math
import numpy as np
import os

fontsize = 18
linewidth = 2
markersize = 8
plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42})
fontsize_legend = 18
color_list = ['#FF1F5B', '#009ADE',  '#F28522', '#58B272', '#AF58BA', '#A6761D','#1f77b4','#ff7f0e']
marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P',  '*','>','<','x']

# 获取一组实验的json文件的路径
def get_json_file_list(dir_path):
    files = []
    for file_name in os.listdir(dir_path):
        _, postfix = os.path.splitext(file_name)
        if postfix == ".json":
            json_file = os.path.join(dir_path, file_name)
            files.append(json_file)
    return files


user_range = (40, 70)
user_step = 3
num_algorithms = 4
def process_data(dir_path):
    json_file_list = get_json_file_list(dir_path)

    costs = {}
    running_times = {}
    for u in range(user_range[0], user_range[1] + user_step, user_step):
        costs[u] = [[] for _ in range(num_algorithms)]
        running_times[u] = [[] for _ in range(num_algorithms)]

    for file in json_file_list:
        raw_data = json.load(open(file))

        raw_cost = raw_data['cost']
        for u_num, cost_arr in raw_cost.items():
            for i in range(num_algorithms):
                costs[int(u_num)][i].extend(cost_arr[i])

        raw_running_time = raw_data['running_time']
        for u_num, running_time_arr in raw_running_time.items():
            for i in range(num_algorithms):
                running_times[int(u_num)][i].extend(running_time_arr[i])

    print(costs)
    print(running_times)

    avg_costs = {}
    avg_running_times = {}
    for u in range(user_range[0], user_range[1] + user_step, user_step):
        avg_costs[u] = []
        avg_running_times[u] = []

    for u_num, arr in costs.items():
        for i in range(num_algorithms):
            avg_c = np.mean(arr[i])
            avg_costs[u_num].append(avg_c)
    for u_num, arr in running_times.items():
        for i in range(num_algorithms):
            avg_r = np.mean(arr[i]) / 1000      # 秒
            avg_running_times[u_num].append(avg_r)

    print("---------------------")
    print("avg_cost: ", avg_costs)
    print("avg_running_time: ", avg_running_times)
    print("---------------------")

    res_cost = [[] for _ in range(num_algorithms)]
    for u_num, c in avg_costs.items():
        for i in range(num_algorithms):
            res_cost[i].append(c[i])
    res_running_time = [[] for _ in range(num_algorithms)]
    for u_num, r in avg_running_times.items():
        for i in range(num_algorithms):
            res_running_time[i].append(r[i])
    print("res_cost: ", res_cost)
    print("res_running_time: ", res_running_time)

    return res_cost, res_running_time

def draw_cost(costs):
    plt.figure()
    plt.ylabel("Average Cost")
    plt.xlabel("Number of Users")
    plt.grid(linestyle='--')
    plt.tight_layout()

    algs = ["Random", "Nearest", "Greedy", "RL"]
    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)
    plt.yticks(ticks=[i for i in range(0, 15000, 1000)])

    # min_y = np.min(cost)
    # max_y = np.max(cost)
    # plt.yticks(np.arange(int(min_y / 10) * 10, math.ceil(max_y / 10) * 10, 50))

    for i in range(4):
        plt.plot(x, costs[i], label=algs[i], color=color_list[i], marker=marker_list[i])

    plt.legend(fontsize=fontsize_legend)
    # plt.savefig(save_file, bbox_inches="tight")
    plt.show()

def draw_running_time(times):
    plt.figure()
    plt.ylabel("Average Running Time(s)")
    plt.xlabel("Number of Users")
    plt.grid(linestyle='--')
    plt.tight_layout()

    algs = ["Random", "Nearest", "Greedy", "RL"]
    x = [i for i in range(user_range[0], user_range[1] + user_step, user_step)]
    plt.xticks(ticks=x)
    # plt.yticks(ticks=[i for i in range(0, 15000, 1000)])

    # min_y = np.min(cost)
    # max_y = np.max(cost)
    # plt.yticks(np.arange(int(min_y / 10) * 10, math.ceil(max_y / 10) * 10, 50))

    for i in range(4):
        plt.plot(x, times[i], label=algs[i], color=color_list[i], marker=marker_list[i])

    plt.legend(fontsize=fontsize_legend)
    # plt.savefig(save_file, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    # raw_data_path = "./5_23_u40-70"
    raw_data_path = "./5_24_u40-70_step3_node40"

    cost, running_time = process_data(raw_data_path)
    draw_cost(cost)
    draw_running_time(running_time)
