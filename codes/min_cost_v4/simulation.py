import random

from codes.min_cost_v4.parameters import environment_configuration
from numpy.random import SeedSequence
import numpy.random
from codes.min_cost_v4.algorithms.random import RandomAssignmentAllocation
from codes.min_cost_v4.algorithms.nearest import NearestAssignmentAllocation
from codes.min_cost_v4.algorithms.greedy import GreedyAssignmentAllocation
from codes.min_cost_v4.env.environment import Environment

from codes.min_cost_v4.ppo.test import RL

import simplejson as json

result = {}

# environment_configuration["num_user"] = 50

print(environment_configuration["delay_limit"])
# print(environment_configuration["num_user"])

base = 100
for i in range(10):
    cost = []

    # env_seed = 666666
    env_seed = 888888

    user_seed = i + base

    # user_seed = 62160
    print("---- user_seed = {} ----".format(user_seed))

    """
        RL
    """
    cost.append(int(RL(env_seed=env_seed, user_seed=user_seed)))

    env_seed = SeedSequence(env_seed)

    """
        random
    """
    env = Environment(environment_configuration, env_seed)
    env.reset_parameters_about_users(user_seed=user_seed)
    random_alg = RandomAssignmentAllocation(env)
    random_alg.run()
    cost.append(int(env.compute_cost(random_alg.assigned_users)))

    """
        nearest
    """
    env = Environment(environment_configuration, env_seed)
    env.reset_parameters_about_users(user_seed=user_seed)
    nearest_alg = NearestAssignmentAllocation(env)
    nearest_alg.run()
    cost.append(int(env.compute_cost(random_alg.assigned_users)))

    """
        greedy
    """
    env = Environment(environment_configuration, env_seed)
    env.reset_parameters_about_users(user_seed=user_seed)
    greedy_alg = GreedyAssignmentAllocation(env)
    greedy_alg.run()
    cost.append(int(env.compute_cost(random_alg.assigned_users)))

    result[user_seed] = cost

print(result)
prefix = "20230517_210039_"
file = "./result/{}_seed{}_node{}_user{}.json".format(prefix, env_seed.entropy, environment_configuration["num_edge_node"], environment_configuration["num_user"])
with open(file, "a") as fjson:
    json.dump(result, fjson)