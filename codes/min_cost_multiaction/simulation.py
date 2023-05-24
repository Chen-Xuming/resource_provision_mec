from codes.min_cost_multiaction.parameters import environment_configuration
from numpy.random import SeedSequence
from codes.min_cost_multiaction.algorithms.random import RandomAssignmentAllocation
from codes.min_cost_multiaction.algorithms.nearest import NearestAssignmentAllocation
from codes.min_cost_multiaction.algorithms.greedy import GreedyAssignmentAllocation
from codes.min_cost_multiaction.env.environment import Environment

from ppo.test import RL

import simplejson as json

result = {}

environment_configuration["num_user"] = 75

for i in range(10):
    cost = []

    env_seed = 1136447707
    user_seed = i
    print("---- user_seed = {} ----".format(user_seed))

    # RL
    cost.append(int(RL(env_seed=env_seed, user_seed=user_seed)))

    env_seed = SeedSequence(env_seed)

    # random
    env = Environment(environment_configuration, env_seed)
    env.reset_parameters_about_users(user_seed=user_seed)
    random_alg = RandomAssignmentAllocation(env)
    random_alg.run()
    cost.append(int(env.compute_cost(random_alg.assigned_users)))

    # nearest
    env = Environment(environment_configuration, env_seed)
    env.reset_parameters_about_users(user_seed=user_seed)
    nearest_alg = NearestAssignmentAllocation(env)
    nearest_alg.run()
    cost.append(int(env.compute_cost(random_alg.assigned_users)))

    # greedy
    env = Environment(environment_configuration, env_seed)
    env.reset_parameters_about_users(user_seed=user_seed)
    greedy_alg = GreedyAssignmentAllocation(env)
    greedy_alg.run()
    cost.append(int(env.compute_cost(random_alg.assigned_users)))

    result[user_seed] = cost

print(result)
file = "./result/multiaction_user{}.json".format(environment_configuration["num_user"])
with open(file, "a") as fjson:
    json.dump(result, fjson)