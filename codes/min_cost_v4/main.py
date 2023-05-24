from codes.min_cost_v4.parameters import environment_configuration
from numpy.random import SeedSequence
from codes.min_cost_v4.algorithms.greedy import GreedyAssignmentAllocation
from codes.min_cost_v4.algorithms.random import RandomAssignmentAllocation
from codes.min_cost_v4.algorithms.nearest import NearestAssignmentAllocation
from codes.min_cost_v4.env.environment import Environment

"""
54171349637842144159007613790275699200
AssertionError: Interactive delay of users (44, 12) is out of limitation.
"""

num_user = 50
for i in range(1):
    # seed = SeedSequence(4558246304207880488366931966567191030)
    # print("entropy={}".format(seed.entropy))

    user_seed = 262588216
    print("---- user_seed = {} ----".format(user_seed))

    env_seed = SeedSequence(888888)
    print("test #{}, entropy={}".format(i + 1, env_seed.entropy))

    env = Environment(environment_configuration, env_seed)
    env.set_users_and_services_by_given_seed(user_seed, num_user=num_user)
    random_alg = RandomAssignmentAllocation(env)
    random_alg.run()

    env = Environment(environment_configuration, env_seed)
    env.set_users_and_services_by_given_seed(user_seed, num_user=num_user)
    nearest_alg = NearestAssignmentAllocation(env)
    nearest_alg.run()

    env = Environment(environment_configuration, env_seed)
    env.set_users_and_services_by_given_seed(user_seed, num_user=num_user)
    greedy_alg = GreedyAssignmentAllocation(env)
    greedy_alg.run()

