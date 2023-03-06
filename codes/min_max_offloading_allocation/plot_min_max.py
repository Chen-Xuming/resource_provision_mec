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
    max_delay = {}
    running_time = {}

    for alg in algorithms:
        max_delay[alg] = {}
        running_time[alg] = {}
        for i in keys:
            max_delay[alg][i] = []
            running_time[alg][i] = []

    for json_file in json_file_list:
        data = json.load(open(json_file))
        for i in keys:
            data_budget = data[str(i)]
            for alg in algorithms:
                max_delay[alg][i].append(data_budget[alg]["max_delay"] * 1000)  # ms
                running_time[alg][i].append(data_budget[alg]["running_time"])
    return max_delay, running_time

def analyse(max_delay, running_time, algorithms, keys):
    result_max_delay = [[] for i in algorithms]
    result_running_time = [[] for i in algorithms]

    for i, alg in enumerate(algorithms):
        for budget in keys:
            result_max_delay[i].append(np.mean(max_delay[alg][budget]))
            result_running_time[i].append(np.mean(running_time[alg][budget]))

    return result_max_delay, result_running_time

def draw_delay(max_delay, algorithms, keys, save_dir):
    plt.figure()
    plt.xlabel("Budget")
    plt.ylabel("Max Interaction Delay (ms)")
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.xticks(ticks=keys)
    min_y = np.min(max_delay)
    max_y = np.max(max_delay)

    plt.yticks(np.arange(int(min_y / 10) * 10, math.ceil(max_y / 10) * 10, 20))

    for i in range(len(algorithms)):
        plt.plot(keys, max_delay[i], label=algorithms[i], color=color_list[i], marker=marker_list[i])
    plt.legend(fontsize=fontsize_legend)
    plt.savefig(save_dir + "/max_delay.png", bbox_inches="tight")
    plt.show()

def draw():
    users = 200

    dir_path = "result/2023-3-2-user_assignment/{}u-10s".format(users)

    step_map = {
        100: 20,
        200: 30,
        300: 40
    }
    start_map = {
        100: 50,
        200: 100,
        300: 150
    }

    steps = 10
    keys = [i for i in range(start_map[users], step_map[users] * steps + start_map[users], step_map[users])]

    assignment_algorithms = ["random", "nearest", "min_tx_tp"]
    allocation_algorithms = ['min_max_greedy']
    algorithms = []
    for assign_alg in assignment_algorithms:
        for alloc_alg in allocation_algorithms:
            algorithms.append("{} + {}".format(assign_alg, alloc_alg))

    file_list = get_json_file_list(dir_path)
    max_delay, running_time = read_data(file_list, keys, algorithms)

    result_max_delay, result_running_time = analyse(max_delay, running_time, algorithms, keys)

    save_dir = "result/2023-3-2-user_assignment/{}u-10s".format(users)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    draw_delay(result_max_delay, algorithms, keys, save_dir)


if __name__ == "__main__":
    draw()
