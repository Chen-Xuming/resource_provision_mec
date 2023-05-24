from codes.min_cost_multiaction.parameters import environment_configuration
from numpy.random import SeedSequence
from codes.min_cost_multiaction.algorithms.random import RandomAssignmentAllocation
from codes.min_cost_multiaction.algorithms.nearest import NearestAssignmentAllocation
from codes.min_cost_multiaction.algorithms.greedy import GreedyAssignmentAllocation
from codes.min_cost_multiaction.env.environment import Environment

"""
54171349637842144159007613790275699200
AssertionError: Interactive delay of users (44, 12) is out of limitation.
"""


for i in range(1):
    # seed = SeedSequence(4558246304207880488366931966567191030)
    # print("entropy={}".format(seed.entropy))

    user_seed = i
    print("---- user_seed = {} ----".format(user_seed))

    env_seed = SeedSequence(1136447707)
    print("test #{}, entropy={}".format(i + 1, env_seed.entropy))

    env = Environment(environment_configuration, env_seed)
    env.reset_parameters_about_users(user_seed=user_seed)
    random_alg = RandomAssignmentAllocation(env)
    random_alg.run()

    env = Environment(environment_configuration, env_seed)
    env.reset_parameters_about_users(user_seed=user_seed)
    nearest_alg = NearestAssignmentAllocation(env)
    nearest_alg.run()

    env = Environment(environment_configuration, env_seed)
    env.reset_parameters_about_users(user_seed=user_seed)
    greedy_alg = GreedyAssignmentAllocation(env)
    greedy_alg.run()



    # algorithm = RandomAssignmentAllocation(env)
    # algorithm.run()

    print("")