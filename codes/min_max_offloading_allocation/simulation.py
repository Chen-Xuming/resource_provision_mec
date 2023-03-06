from numpy.random import default_rng, SeedSequence
from environment import *

from allocation import *

import copy  # for deepcopy
import os
import json
import simplejson

"""
    Environment配置
"""
num_instance = 1
num_service = 10

env_config = {
    "num_instance": num_instance,
    "num_service_a": num_service,
    "num_service_r": num_service,
    "budget_addition": None,
    "num_group_per_instance": 20,
    "num_user_per_group": 10,
    "min_arrival_rate": 10,
    "max_arrival_rate": 15,
    "min_service_rate_a": 100,
    "max_service_rate_a": 200,
    "min_service_rate_q": 200,
    "max_service_rate_q": 500,
    "min_service_rate_r": 40,
    "max_service_rate_r": 50,
    "min_price": 1,
    "max_price": 5,
    "trigger_probability": 0.2,
    "tx_ua_min": 5,
    "tx_ua_max": 20,
    "tx_aq_min": 5,
    "tx_aq_max": 15,
    "tx_qr_min": 5,
    "tx_ru_min": 15,
    "tx_qr_max": 5,
    "tx_ru_max": 20,
    "assignment_algorithm": None
}

def set_assignment_algorithm(alg: str):
    env_config["assignment_algorithm"] = alg

assignment_algorithms = ["random", "nearest", "min_tx_tp"]

# allocation_algorithms = ['min_max_greedy', 'min_max_equal_weight', 'min_max_surrogate_relaxation', 'min_max_pulp']
allocation_algorithms = ['min_max_greedy']


simulate_user_count = env_config["num_group_per_instance"] * env_config["num_user_per_group"]
result_dir = "result/2023-3-2-user_assignment/{}u-10s".format(simulate_user_count)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

# 如果要使用之前实验的随机种子，那么要进行读取
def get_entropy(dir_path):
    entropy_list = []
    if dir_path is None:
        return entropy_list
    for file_name in os.listdir(dir_path):
        _, postfix = os.path.splitext(file_name)
        if postfix != '.json':
            continue
        json_file = os.path.join(dir_path, file_name)
        data = json.load(open(json_file))
        entropy = data["entropy"]
        entropy_list.append(entropy)
    return entropy_list

# entropy_files = "result/2022-9-14-addition1/{}u-10s".format(simulate_user_count)
# seed_sequence_list = get_entropy(entropy_files)
#
# # 如果某些实验已经跑过了，那就无需再跑
# result_finished_entropy = get_entropy(result_dir)
# start_no = 9
# end_no = start_no + 1
# seed_sequence_list = seed_sequence_list[start_no : end_no]

simulate_no = 14  # simulate编号，方便在控制台查看是哪个实验
print("simulate_no: {}".format(simulate_no))
print("user_num: {}".format(simulate_user_count))
print("assignment_algorithms: {}\n".format(assignment_algorithms))
print("allocation_algorithms: {}\n".format(allocation_algorithms))

simulation_num = 6

"""
    每次增加的budget
    100users ==> 20  (20-200)
    200users ==> 30  (30-300)
    300users ==> 40  (40-400)
"""
budget_addition_each_time = {
    100: 20,
    200: 30,
    300: 40
}
initial_budgets = {
    100: 50,
    200: 100,
    300: 150
}
budget_unit = budget_addition_each_time[simulate_user_count]
initial_budget = initial_budgets[simulate_user_count]

for n in range(simulation_num):
    # # 仅在使用以往实验的随机种子时需要
    # if seed_sequence_list[n] in result_finished_entropy:
    #     continue
    # seed_sequence = SeedSequence(seed_sequence_list[n])

    # 第一跑实验时需要
    seed_sequence = SeedSequence()

    suffix = str(seed_sequence.entropy)[-8:]
    filename_result = result_dir + "/result_{}.json".format(suffix)

    print("========================= simulation: {},  seed_sequence: {} =====================".format(n, seed_sequence.entropy))

    env_config["budget_addition"] = initial_budget

    res_summary = dict()
    res_summary["entropy"] = seed_sequence.entropy

    for _ in range(10):
        print("++++++++++++++++++ budget: {} +++++++++++++++++".format(env_config["budget_addition"]))
        res_summary[env_config["budget_addition"]] = {}

        for assign_alg in assignment_algorithms:
            for alloc_alg in allocation_algorithms:
                comb_alg_name = "{} + {}".format(assign_alg, alloc_alg)
                print("------------ algorithm: {} ----------".format(comb_alg_name))

                rng = default_rng(seed_sequence)
                env_config["assignment_algorithm"] = assign_alg
                env = Env(env_config, rng, seed_sequence)
                cost = env.compute_cost()
                env._cost_budget = cost + env._budget_addition

                res_summary[env_config["budget_addition"]]["num_user"] = env._num_user
                # res_summary[env_config["budget_addition"]]["num_service"] = env._num_service

                allocation_class = eval(alloc_alg)
                algorithm = allocation_class(env)
                algorithm.set_num_server()
                res_summary[env_config["budget_addition"]][comb_alg_name] = algorithm.get_result_dict()

        env_config["budget_addition"] += budget_unit

    # 写入json
    with open(filename_result, "a") as fid_json:
        simplejson.dump(res_summary, fid_json)








