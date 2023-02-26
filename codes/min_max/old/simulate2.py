from numpy.random import default_rng, SeedSequence
from codes.old.env2 import *

# 只考虑排队时延的算法
# from algorithm2 import *

# 考虑整个交互时延的算法


import copy  # for deepcopy
import os
import json
import simplejson


description = "SURROGATE 300USERS high_a * 2"
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
    "num_service_b": num_service,
    "num_service_c": num_service,
    "num_service_r": num_service,
    "budget_addition": 10,
    "num_group_per_instance": 30,
    "num_user_per_group": 10,
    "min_arrival_rate": 10,
    "max_arrival_rate": 15,
    "min_service_rate": 500,        # 500
    "max_service_rate": 600,        # 600
    "min_price": 1,
    "max_price": 5,
    "trigger_probability": 0.2,
    "min_tx": 5,       # 传输时延
    "max_tx": 10,
}

# algorithm_name_list = ["min_max_pulp"]
# algorithm_name_list = ['min_max_greedy', 'min_max_equal_weight', 'min_max_surrogate_relaxation', 'min_max_pulp']
# algorithm_name_list = ['min_max_greedy', 'min_max_equal_weight', 'min_max_equal_weight_dp']
# algorithm_name_list = ['min_max_surrogate_relaxation', 'min_max_greedy', 'min_max_equal_weight']
algorithm_name_list = ['min_max_surrogate_relaxation']


simulate_user_count = env_config["num_group_per_instance"] * env_config["num_user_per_group"]
result_dir = "result/2022-10-4-surrogate-double_high_a/{}u-10s".format(simulate_user_count)
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

entropy_files = "result/2022-9-14-addition1/{}u-10s".format(simulate_user_count)
seed_sequence_list = get_entropy(entropy_files)

# 如果某些实验已经跑过了，那就无需再跑
result_finished_entropy = get_entropy(result_dir)
start_no = 9
end_no = start_no + 1
seed_sequence_list = seed_sequence_list[start_no : end_no]


simulate_no = 9 # simulate编号，方便在控制台查看是哪个实验
print("simulate_no: {}".format(simulate_no))
print("user_num: {}".format(simulate_user_count))
print("algorithms: {}\n".format(algorithm_name_list))

simulation_num = 1

"""
    每次增加的budget
    100users ==> 25  (25-250)
    200users ==> 30  (30-300)
    300users ==> 35  (35-350)
"""
budget_options = {
    100: 25,
    200: 30,
    300: 35
}
budget_unit = budget_options[simulate_user_count]

for n in range(simulation_num):
    # 仅在使用以往实验的随机种子时需要
    if seed_sequence_list[n] in result_finished_entropy:
        continue
    seed_sequence = SeedSequence(seed_sequence_list[n])

    # 第一跑实验时需要
    # seed_sequence = SeedSequence()

    suffix = str(seed_sequence.entropy)[-8:]
    filename_result = result_dir + "/result_{}.json".format(suffix)

    print("------------- simulation: {},  seed_sequence: {} -------------".format(n, seed_sequence))

    """
        如果规模env规模太大，那么初始化的时间会很长。
        对于只有budget_addition不同的env，它们只有_budget_addition, _cost_budget不同。
        可以初始化一个 budget_addition = 10 的env, 然后在不同budget_addition的实验中通过 deep_copy 拷贝，并手动改变上述两个值，这样可以大幅减少运行时间
    """
    rng = default_rng(seed_sequence)
    env_config["budget_addition"] = budget_unit
    env_for_copy = Env(env_config, rng, seed_sequence)

    res_summary = {}
    res_summary["entropy"] = seed_sequence.entropy

    # 每次仿真把预算增加10(10-100)
    budget_addition = budget_unit
    for b in range(10):
        print("--------- budget: {} ----------".format(budget_addition))
        res_summary[budget_addition] = {}

        rng = default_rng(seed_sequence)
        st = rng.bit_generator.state

        for name in algorithm_name_list:
            print("--------- algorithm: {} ----------".format(name))

            """
                env 深拷贝和重置
            """
            env = copy.deepcopy(env_for_copy)
            env._budget_addition = budget_addition
            cost = env.compute_cost()
            env._cost_budget = cost + env._budget_addition

            res_summary[budget_addition]["num_user"] = env._num_user
            res_summary[budget_addition]["num_service"] = env._num_service

            algorithm_class = eval(name)
            algorithm = algorithm_class(env)
            algorithm.set_num_server()
            res_summary[budget_addition][name] = algorithm.get_result_dict()
            rng.bit_generator.state = st
        budget_addition += budget_unit

    # 写入json
    with open(filename_result, "a") as fid_json:
        simplejson.dump(res_summary, fid_json)






































