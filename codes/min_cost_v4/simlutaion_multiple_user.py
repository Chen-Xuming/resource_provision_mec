from numpy import random

import sys
sys.path.append("F:/resource_provision_mec")

from codes.min_cost_v4.parameters import environment_configuration
from numpy.random import SeedSequence
from codes.min_cost_v4.algorithms.random import RandomAssignmentAllocation
from codes.min_cost_v4.algorithms.nearest import NearestAssignmentAllocation
from codes.min_cost_v4.algorithms.greedy import GreedyAssignmentAllocation
from codes.min_cost_v4.env.environment import Environment

from codes.min_cost_v4.ppo.test import RL

import simplejson as json




print(environment_configuration["delay_limit"])
print("delay_limit = ", environment_configuration["delay_limit"])

env_seed = 888888

simulation_no = 9   # 文件号

# 用户数及测试次数
user_range = (40, 70)
user_range_step = 3
simulation_times_each_num_user = 1

print("===============================")
print("env_seed: ", env_seed)
print("num_edge_node: ", environment_configuration["num_edge_node"])
print("user_range: ", user_range)
print("user_step: ", user_range_step)
print("simulation_no: ", simulation_no)
print("===============================")


# 测试算法
algorithms = ["Random", "Nearest", "Greedy", "RL"]

results = {
    "cost": {},
    "running_time": {}
}

for num_user in range(user_range[0], user_range[1] + user_range_step, user_range_step):
    res_costs = [[] for _ in range(len(algorithms))]
    res_running_time = [[] for _ in range(len(algorithms))]

    for i in range(simulation_times_each_num_user):
        user_seed = random.randint(0, 10000000)
        env_seed_sequence = SeedSequence(env_seed)

        for j, alg_name in enumerate(algorithms):
            if alg_name == "Random":
                env = Environment(environment_configuration, env_seed_sequence)
                env.set_users_and_services_by_given_seed(user_seed=user_seed, num_user=num_user)
                random_alg = RandomAssignmentAllocation(env)
                random_alg.run()
                res_costs[j].append(int(env.compute_cost()))
                res_running_time[j].append(random_alg.get_running_time())

            elif alg_name == "Nearest":
                env = Environment(environment_configuration, env_seed_sequence)
                env.set_users_and_services_by_given_seed(user_seed=user_seed, num_user=num_user)
                nearest_alg = NearestAssignmentAllocation(env)
                nearest_alg.run()
                res_costs[j].append(int(env.compute_cost()))
                res_running_time[j].append(nearest_alg.get_running_time())

            elif alg_name == "Greedy":
                env = Environment(environment_configuration, env_seed_sequence)
                env.set_users_and_services_by_given_seed(user_seed=user_seed, num_user=num_user)
                greedy_alg = GreedyAssignmentAllocation(env)
                greedy_alg.run()
                res_costs[j].append(int(env.compute_cost()))
                res_running_time[j].append(greedy_alg.get_running_time())

            elif alg_name == "RL":
                cost, running_time = RL(env_seed_sequence, user_seed, num_user)
                res_costs[j].append(int(cost))
                res_running_time[j].append(running_time)

        print("---------------------")
        print("num_user = {}, simulation #{}".format(num_user, i))
        for j in range(len(algorithms)):
            print("algorithm: {}, cost = {}, running_time = {}".format(algorithms[j], res_costs[j][i], res_running_time[j][i]))
        print("----------------------")

    results["cost"][num_user] = res_costs
    results["running_time"][num_user] = res_running_time

print(results)

# file = "./result/5_23_u40-70_v2/seed{}_node{}_user{}-{}_{}.json".format(env_seed, environment_configuration["num_edge_node"], user_range[0], user_range[1], simulation_no)
file = "./result/5_24_u40-70_step3_node40/seed{}_node{}_user{}-{}_{}.json".format(env_seed, environment_configuration["num_edge_node"], user_range[0], user_range[1], simulation_no)
with open(file, "a") as fjson:
    json.dump(results, fjson)
