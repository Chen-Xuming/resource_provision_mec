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
plt.rcParams.update(
    {'font.size': fontsize, 'lines.linewidth': linewidth, 'lines.markersize': markersize, 'pdf.fonttype': 42,
     'ps.fonttype': 42})
fontsize_legend = 18
color_list = ['#FF1F5B', '#009ADE', '#F28522', '#58B272', '#AF58BA', '#A6761D', '#1f77b4', '#ff7f0e']
marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P', '*', '>', '<', 'x']


# 获取一组实验的json文件的路径
def get_json_file_list(dir_path):
    files = []
    for file_name in os.listdir(dir_path):
        _, postfix = os.path.splitext(file_name)
        if postfix == ".json":
            json_file = os.path.join(dir_path, file_name)
            files.append(json_file)
    return files

"""
    读取json数据
    
    源数据格式：
    {
        "entropy": 272679763379360388665259216836132909102,
        "30": {
            "num_user": 200,
            "num_service": 142,
            "min_max_greedy": {
                "max_delay": 0.14649141663707477,
                "min_delay": 0.06431527959813121,
                "avg_delay": 0.09057172379086921,
                "min_max_difference": 0.08217613703894355,
                "running_time": 22.0186710357666
            },
            "min_max_equal_weight": {
                "max_delay": 0.3288780126104598,
                "min_delay": 0.060590815906243274,
                "avg_delay": 0.08887598017487233,
                "min_max_difference": 0.2682871967042165,
                "running_time": 18.014907836914062
            }
        },
        "60": {
            "num_user": 200,
            "num_service": 142,
            "min_max_greedy": {
                "max_delay": 0.12583968094031828,
                "min_delay": 0.06431527959813121,
                "avg_delay": 0.08533268355027505,
                "min_max_difference": 0.06152440134218706,
                "running_time": 37.029266357421875
            },
            "min_max_equal_weight": {
                "max_delay": 0.12510406181231457,
                "min_delay": 0.056581164112970705,
                "avg_delay": 0.0755004987040882,
                "min_max_difference": 0.06852289769934386,
                "running_time": 38.030385971069336
            }
        },
        ...
    }
"""
def read_data(json_file_list, keys, algorithms):
    max_delay = {}
    min_delay = {}
    avg_delay = {}
    min_max_difference = {}

    for alg in algorithms:
        max_delay[alg] = {}
        min_delay[alg] = {}
        avg_delay[alg] = {}
        min_max_difference[alg] = {}
        for i in keys:
            max_delay[alg][i] = []
            min_delay[alg][i] = []
            avg_delay[alg][i] = []
            min_max_difference[alg][i] = []

    for json_file in json_file_list:
        data = json.load(open(json_file))
        for i in keys:
            data_budget = data[str(i)]
            for alg in algorithms:
                max_delay[alg][i].append(data_budget[alg]["max_delay"] * 1000)  # ms
                min_delay[alg][i].append(data_budget[alg]["min_delay"] * 1000)  # ms
                avg_delay[alg][i].append(data_budget[alg]["avg_delay"] * 1000)  # ms
                min_max_difference[alg][i].append(data_budget[alg]["min_max_difference"] * 1000)  # ms

    return max_delay, min_delay, avg_delay, min_max_difference

def analyse(max_delay, min_delay, avg_delay, min_max_difference, algorithms, keys):
    result_max_delay = [[] for _ in algorithms]
    result_min_delay = [[] for _ in algorithms]
    result_avg_delay = [[] for _ in algorithms]
    result_min_max_difference = [[] for _ in algorithms]

    for i, alg in enumerate(algorithms):
        for budget in keys:
            result_max_delay[i].append(np.mean(max_delay[alg][budget]))
            result_min_delay[i].append(np.mean(min_delay[alg][budget]))
            result_avg_delay[i].append(np.mean(avg_delay[alg][budget]))
            result_min_max_difference[i].append(np.mean(min_max_difference[alg][budget]))

    return result_max_delay, result_min_delay, result_avg_delay, result_min_max_difference

def draw_max_min_avg(max_delay, min_delay, avg_delay, algorithm, keys, save_dir):
    plt.figure()
    plt.xlabel("Budget")
    plt.ylabel("Max-Avg-Min Delay (ms)")
    plt.grid(linestyle='--')
    plt.tight_layout()
    plt.xticks(ticks=keys)
    min_y = np.min(min_delay)
    max_y = np.max(max_delay)

    plt.yticks(np.arange(int(min_y / 10) * 10, math.ceil(max_y / 10) * 10, 35))

    # fig_title = ""
    # if algorithm == "min_max_greedy":
    #     fig_title = "Greedy"
    # if algorithm == "min_max_equal_weight":
    #     fig_title = "Equal Weight"

    # plt.title(fig_title)
    plt.title(algorithm)

    plt.plot(keys, max_delay, label="max_delay", color=color_list[0], marker=marker_list[0])
    plt.plot(keys, avg_delay, label="avg_delay", color=color_list[1], marker=marker_list[1])
    plt.plot(keys, min_delay, label="min_delay", color=color_list[2], marker=marker_list[2])

    plt.legend(fontsize=fontsize_legend)

    plt.savefig(save_dir + "/max_avg_min_{}.png".format(algorithm), bbox_inches="tight")
    plt.show()


def draw_min_max_delay_difference(max_min_delay_difference, algorithms, keys, save_dir):
    print(max_min_delay_difference)

    from matplotlib.patches import Patch

    width = 3
    offset_w = 0.5
    n = len(max_min_delay_difference)
    if n % 2 == 0:
        offset = np.arange(1 - n, n + 1, 2) * width * offset_w
    else:
        offset = np.arange(-n + 1, n + 1, 2) * width * offset_w

    plt.figure()
    plt.xlabel("Budget")
    plt.ylabel("Delay Difference (ms)")

    plt.title("Difference Between Max & Min Delay")

    plt.tight_layout()
    plt.yticks(np.arange(int(10 / 10) * 10, math.ceil(600 / 10) * 10, 40))
    plt.xticks(ticks=keys)

    for i, alg in enumerate(algorithms):
        for j in range(len(keys)):
            plt.bar(keys[j] + offset[i], max_min_delay_difference[i][j], width=width, color=color_list[i])

    # algorithms = ["greedy", "equal weight"]
    patches = [Patch(facecolor=color_list[i], label=alg) for i, alg in enumerate(algorithms)]

    plt.legend(fontsize=fontsize_legend, handles=patches)
    plt.grid(linestyle='--')
    plt.savefig(save_dir + "/max_min_delay_difference.png", bbox_inches="tight")
    plt.show()

def draw_result_fig():
    users = 200

    dir_path = "result/2023-2-17-min_diff_3_algs/{}u-10s".format(users)

    step_map = {
        100: 20,
        200: 30,
        300: 40
    }
    keys = [i for i in range(step_map[users], step_map[users] * 10 + step_map[users], step_map[users])]

    algorithms = ['min_max_greedy', 'min_max_equal_weight', 'delay_balance']

    file_list = get_json_file_list(dir_path)
    max_delay, min_delay, avg_delay, min_max_difference = read_data(file_list, keys, algorithms)

    result_max_delay, result_min_delay, result_avg_delay, result_min_max_difference = analyse(max_delay, min_delay, avg_delay, min_max_difference, algorithms, keys)

    save_dir = "result/2023-2-17-min_diff_3_algs/{}u-10s".format(users)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for i, alg in enumerate(algorithms):
        draw_max_min_avg(result_max_delay[i], result_min_delay[i], result_avg_delay[i], alg, keys, save_dir)

    draw_min_max_delay_difference(result_min_max_difference, algorithms, keys, save_dir)

if __name__ == "__main__":
    draw_result_fig()
