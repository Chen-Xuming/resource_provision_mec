from codes.min_cost.parameters import environment_configuration
from numpy.random import SeedSequence
from codes.min_cost.algorithms.greedy import GreedyAssignmentAllocation
from codes.min_cost.algorithms.random import RandomAssignmentAllocation
from codes.min_cost.algorithms.nearest import NearestAssignmentAllocation
from codes.min_cost.env.environment import Environment

"""
54171349637842144159007613790275699200
AssertionError: Interactive delay of users (44, 12) is out of limitation.
"""


for i in range(1):
    # seed = SeedSequence(4558246304207880488366931966567191030)
    # print("entropy={}".format(seed.entropy))

    # seed = SeedSequence(123456789)
    seed = SeedSequence(987654321)
    print("test #{}, entropy={}".format(i + 1, seed.entropy))

    env = Environment(environment_configuration, seed)
    random_alg = RandomAssignmentAllocation(env)
    random_alg.run()

    env = Environment(environment_configuration, seed)
    nearest_alg = NearestAssignmentAllocation(env)
    nearest_alg.run()

    env = Environment(environment_configuration, seed)
    greedy_alg = GreedyAssignmentAllocation(env)
    greedy_alg.run()



    # algorithm = RandomAssignmentAllocation(env)
    # algorithm.run()

    print("")