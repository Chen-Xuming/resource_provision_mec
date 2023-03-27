import os
import json
import numpy as np
import matplotlib.pyplot as plt
import math

"""
    style of figures
"""
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


def read_data(json_file_list, keys, algorithms):
    cost = {}
    running_time = {}
    avg_delay = {}

    for alg in algorithms:
        cost[alg] = {}
        running_time[alg] = {}
        avg_delay[alg] = {}
        for i in keys:
            cost[alg][i] = []
            running_time[alg][i] = []
            avg_delay[alg][i] = []

    for json_file in json_file_list:
        data = json.load(open(json_file))
        for i in keys:
            data_i = data[str(i)]
            for alg in algorithms:
                cost[alg][i].append(data_i[alg]["cost"])
                avg_delay[alg][i].append(data_i[alg]["avg_delay"] * 1000)  # ms
                running_time[alg][i].append(data_i[alg]["running_time"] / 1000)  # s

    return cost, avg_delay, running_time

def analyse(cost, avg_delay, running_time, algorithms, keys):
    result_cost = [[] for i in algorithms]
    result_avg_delay = [[] for i in algorithms]
    result_running_time = [[] for i in algorithms]

    for i, alg in enumerate(algorithms):
        for key in keys:
            result_cost[i].append(np.mean(cost[alg][key]))
            result_avg_delay[i].append(np.mean(avg_delay[alg][key]))
            result_running_time[i].append(np.mean(running_time[alg][key]))

    return result_cost, result_avg_delay, result_running_time


def draw_cost(cost, algorithms, keys, key_name, save_dir):
    plt.figure()
    plt.xlabel(key_name)
    plt.ylabel("Cost")
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.xticks(ticks=keys)
    min_y = np.min(cost)
    max_y = np.max(cost)

    # plt.yticks(np.arange(int(min_y / 10) * 10, math.ceil(max_y / 10) * 10, 50))

    for i in range(len(algorithms)):
        plt.plot(keys, cost[i], label=algorithms[i], color=color_list[i], marker=marker_list[i])
    plt.legend(fontsize=fontsize_legend)
    plt.savefig(save_dir + "/cost.png", bbox_inches="tight")
    plt.show()

def draw_avg_delay(avg_delay, algorithms, keys, key_name, save_dir):
    plt.figure()
    plt.xlabel(key_name)
    plt.ylabel("Average Interactive Delay(ms)")
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.xticks(ticks=keys)
    min_y = np.min(avg_delay)
    max_y = np.max(avg_delay)

    # plt.yticks(np.arange(int(min_y / 10) * 10, math.ceil(max_y / 10) * 10, 10))

    for i in range(len(algorithms)):
        plt.plot(keys, avg_delay[i], label=algorithms[i], color=color_list[i], marker=marker_list[i])
    plt.legend(fontsize=fontsize_legend)
    plt.savefig(save_dir + "/avg_delay.png", bbox_inches="tight")
    plt.show()


def draw_running_time(running_time, algorithms, keys, key_name, save_dir):
    plt.figure()
    plt.xlabel(key_name)
    plt.ylabel("Running Time(s)")
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.xticks(ticks=keys)
    min_y = np.min(running_time)
    max_y = np.max(running_time)
    plt.ylim([min_y - 100, max_y + 100])
    for i in range(len(algorithms)):
        plt.plot(keys, running_time[i], label=algorithms[i], color=color_list[i], marker=marker_list[i])
    plt.legend(fontsize=fontsize_legend)
    plt.savefig(save_dir + "/running_time.png", bbox_inches="tight")
    plt.show()


def draw_fix_node_vary_user():
    num_node = 50

    dir_path = "./result/20230323/fix_{}node".format(num_node)

    key_name = "Number of User"
    keys = [i for i in range(60, 100 + 10, 10)]

    algorithms = ["min_cost_random", "min_cost_nearest", "min_cost_greedy"]

    file_list = get_json_file_list(dir_path)

    cost, avg_delay, running_time = read_data(file_list, keys, algorithms)

    result_cost, result_avg_delay, result_running_time = analyse(cost, avg_delay, running_time, algorithms, keys)

    save_dir = dir_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    draw_cost(result_cost, algorithms, keys, key_name, save_dir)
    draw_avg_delay(result_avg_delay, algorithms, keys, key_name, save_dir)
    draw_running_time(result_running_time, algorithms, keys, key_name, save_dir)

def draw_fix_user_vary_node():
    num_user = 100

    dir_path = "./result/20230323/fix_{}user".format(num_user)

    key_name = "Number of Edge-node"
    keys = [i for i in range(40, 60 + 5, 5)]

    algorithms = ["min_cost_random", "min_cost_nearest", "min_cost_greedy"]

    file_list = get_json_file_list(dir_path)

    cost, avg_delay, running_time = read_data(file_list, keys, algorithms)

    result_cost, result_avg_delay, result_running_time = analyse(cost, avg_delay, running_time, algorithms, keys)

    save_dir = dir_path
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    draw_cost(result_cost, algorithms, keys, key_name, save_dir)
    draw_avg_delay(result_avg_delay, algorithms, keys, key_name, save_dir)
    draw_running_time(result_running_time, algorithms, keys, key_name, save_dir)

if __name__ == '__main__':
    # draw_fix_node_vary_user()

    draw_fix_user_vary_node()