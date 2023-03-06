from numpy.random import default_rng, SeedSequence
from environment import *

# 只考虑排队时延的算法
from algorithm import min_max_pulp

# 考虑整个交互时延的算法


import copy  # for deepcopy
import os
import json
import simplejson

import sys

args = sys.argv[1:]  # 从第二个元素开始取，因为第一个元素是程序名称本身
print("args = {}".format(args))

simulate_no = int(args[0])  # simulate编号，方便在控制台查看是哪个实验
num_group_per_instance = int(args[1])


description = ""
print("************************************")
print(description)
print("************************************")

"""
    Environment配置
"""
num_instance = 1
num_service = 10
env_config = {
    "num_instance": num_instance,
    "num_service_a": num_service,
    "num_service_r": num_service,
    "budget_addition": 20,
    "num_group_per_instance": num_group_per_instance,
    "num_user_per_group": 10,
    "min_arrival_rate": 10,
    "max_arrival_rate": 15,
    "min_max_service_rate_a": (100, 200),
    "min_max_service_rate_q": (200, 500),
    "min_max_service_rate_r": (40, 50),
    "min_price": 1,
    "max_price": 5,
    "trigger_probability": 0.2,
    "tx_ua_min": 4,
    "tx_ua_max": 6,
    "tx_aq_min": 2,
    "tx_aq_max": 4,
    "tx_qr_min": 2,
    "tx_qr_max": 4,
    "tx_ru_min": 4,
    "tx_ru_max": 6
}

simulate_user_count = env_config["num_group_per_instance"] * env_config["num_user_per_group"]
result_dir = "result/PULP_solutions".format(simulate_user_count)
if not os.path.exists(result_dir):
    os.makedirs(result_dir)

print("simulate_no: {}".format(simulate_no))
print("user_num: {}".format(simulate_user_count))

simulation_num = 10

"""
    budget范围（从初始值开始，每次增加若干）
    100users ==> 50  (50-230)   step = 20
    200users ==> 50  (50-320)   step = 30
    300users ==> 50  (50-410)   step = 40
"""
initial_budgets = {
    100: 50,
    200: 50,
    300: 50
}
budget_steps = {
    100: 20,
    200: 30,
    300: 40
}
budget_step = budget_steps[simulate_user_count]
initial_budget = initial_budgets[simulate_user_count]


"""
    pulp time_limit
"""
pulp_time_limit_list = {
    100: 1800,      # 30min
    200: 3600,      # 1h
    300: 3600
}
pulp_time_limit = pulp_time_limit_list[simulate_user_count]


for n in range(simulation_num):
    # 第一跑实验时需要
    seed_sequence = SeedSequence()

    suffix = str(seed_sequence.entropy)[-8:]
    filename_result = result_dir + "/pulp_{}.json".format(suffix)

    print("------------- simulation: {},  seed_sequence: {} -------------".format(n, seed_sequence))

    """
        如果规模env规模太大，那么初始化的时间会很长。
        对于只有budget_addition不同的env，它们只有_budget_addition, _cost_budget不同。
        可以初始化一个 budget_addition = 10 的env, 然后在不同budget_addition的实验中通过 deep_copy 拷贝，并手动改变上述两个值，这样可以大幅减少运行时间
    """
    rng = default_rng(seed_sequence)
    env_config["budget_addition"] = initial_budget
    env_for_copy = Env(env_config, rng, seed_sequence)

    env_info = env_for_copy.get_env_info()

    res_summary = dict()

    res_summary["environment_configuration"] = env_info
    # res_summary["entropy"] = seed_sequence.entropy

    # 每次仿真把预算增加budget_unit
    budget_addition = initial_budget
    for b in range(10):
        print("--------- budget: {} ----------".format(budget_addition))
        res_summary[budget_addition] = {}

        rng = default_rng(seed_sequence)
        st = rng.bit_generator.state

        # ---- 求解 -----
        # env 深拷贝和重置
        env = copy.deepcopy(env_for_copy)
        env._budget_addition = budget_addition
        cost = env.compute_cost()
        env._cost_budget = cost + env._budget_addition

        pulp_solver = min_max_pulp(env)
        pulp_solver.set_time_limit(time_limit=pulp_time_limit)
        solution = pulp_solver.set_num_server()
        result_dict = pulp_solver.get_result_dict()
        result_dict["solution"] = solution
        res_summary[budget_addition] = result_dict
        rng.bit_generator.state = st
        budget_addition += budget_step


    # 写入json
    with open(filename_result, "w") as fid_json:
        simplejson.dump(res_summary, fid_json)