import simplejson as json
from matplotlib import pyplot as plt
import math
import numpy as np

fontsize = 18
linewidth = 2
markersize = 8
plt.rcParams.update({'font.size':fontsize, 'lines.linewidth':linewidth, 'lines.markersize':markersize, 'pdf.fonttype':42, 'ps.fonttype':42})
fontsize_legend = 18
color_list = ['#FF1F5B', '#009ADE',  '#F28522', '#58B272', '#AF58BA', '#A6761D','#1f77b4','#ff7f0e']
marker_list = ['o', '^', 'X', 'd', 's', 'v', 'P',  '*','>','<','x']

user_num = 75
data_file = "./result/unshared_a_user{}.json".format(user_num)
save_file = "./result/unshared_a_user{}.svg".format(user_num)

def draw():
    cost = [[] for i in range(4)]
    data = json.load(open(data_file))
    for key, value in data.items():
        for i in range(4):
            cost[i].append(value[i])
    print(cost)

    avg_cost = []
    for arr in cost:
        avg_cost.append(np.mean(arr))

    print(avg_cost)

    plt.figure()
    plt.ylabel("Average Cost")
    plt.xlabel("num_user={}".format(user_num))
    # plt.title("num_user={}, num_edge_node={}".format(user_num, 25))

    algs = ["RL", "Random", "Nearest", "Greedy"]
    for i in range(4):
        plt.bar(algs[i], avg_cost[i], width=0.8)

    for a, b in zip(algs, avg_cost):  # 柱子上的数字显示
        plt.text(a, b, b, ha='center', va='bottom', fontsize=14)

    plt.savefig(save_file, bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    draw()