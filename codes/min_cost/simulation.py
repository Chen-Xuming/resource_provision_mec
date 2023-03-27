import simplejson
from numpy.random import default_rng, SeedSequence
from env_environment import *

from alg_random import RandomAssignmentAllocation
from alg_greedy import GreedyAssignmentAllocation
from alg_nearest import NearestAssignmentAllocation

from parameters import environment_configuration, show_env_config

import os
import simplejson as json

algorithm_name_to_class = {
    "min_cost_greedy": "GreedyAssignmentAllocation",
    "min_cost_random": "RandomAssignmentAllocation",
    "min_cost_nearest": "NearestAssignmentAllocation"
}

algorithm_list = ["min_cost_random", "min_cost_nearest", "min_cost_greedy"]
# algorithm_list = ["min_cost_random", "min_cost_nearest"]


"""
    固定节点数，改变用户数量
"""
def fix_num_node_vary_num_user():

    environment_configuration["num_edge_node"] = 50
    num_user_range = (60, 100, 10)

    saving_path = "./result/20230323/fix_{}node".format(environment_configuration["num_edge_node"])
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    seed = SeedSequence()
    suffix = str(seed.entropy)[-8:]
    saving_file = saving_path + "/{}.json".format(suffix)

    print("seed.entropy = {}".format(seed.entropy))

    res_summary = dict()
    res_summary["entropy"] = seed.entropy

    for num_user in range(num_user_range[0], num_user_range[1] + num_user_range[2], num_user_range[2]):
        environment_configuration["num_user"] = num_user
        print("--------------- num_user = {} ------------".format(num_user))

        res_summary[num_user] = {}

        for algorithm_name in algorithm_list:
            print("--------------- algorithm: {} -----------".format(algorithm_name))

            env = Environment(environment_configuration, seed)

            algorithm_class = eval(algorithm_name_to_class[algorithm_name])
            algorithm = algorithm_class(env)
            algorithm.run()

            res_summary[num_user][algorithm_name] = algorithm.get_results()

    with open(saving_file, 'a') as fjson:
        json.dump(res_summary, fjson)

"""
    固定用户个数，改变节点个数
"""
def fix_num_user_vary_num_node():
    environment_configuration["num_user"] = 100
    num_node_range = (40, 60, 5)

    saving_path = "./result/20230323/fix_{}user".format(environment_configuration["num_user"])
    if not os.path.exists(saving_path):
        os.makedirs(saving_path)

    seed = SeedSequence()
    suffix = str(seed.entropy)[-8:]
    saving_file = saving_path + "/{}.json".format(suffix)

    print("seed.entropy = {}".format(seed.entropy))

    res_summary = dict()
    res_summary["entropy"] = seed.entropy

    for num_node in range(num_node_range[0], num_node_range[1] + num_node_range[2], num_node_range[2]):
        environment_configuration["num_edge_node"] = num_node
        print("--------------- num_node = {} ------------".format(num_node))

        res_summary[num_node] = {}

        for algorithm_name in algorithm_list:
            print("--------------- algorithm: {} -----------".format(algorithm_name))

            env = Environment(environment_configuration, seed)

            algorithm_class = eval(algorithm_name_to_class[algorithm_name])
            algorithm = algorithm_class(env)
            algorithm.run()

            res_summary[num_node][algorithm_name] = algorithm.get_results()

    with open(saving_file, 'a') as fjson:
        json.dump(res_summary, fjson)



if __name__ == '__main__':
    for i in range(2):
        print("=============================== simulation {} =============================".format(i))
        # fix_num_node_vary_num_user()

        fix_num_user_vary_num_node()

