from codes.min_cost.parameters import environment_configuration
from numpy.random import SeedSequence
from codes.min_cost.alg_random import RandomAssignmentAllocation
from codes.min_cost.alg_nearest import NearestAssignmentAllocation
from codes.min_cost.alg_greedy import GreedyAssignmentAllocation
from codes.min_cost.env_environment import Environment

"""
54171349637842144159007613790275699200
AssertionError: Interactive delay of users (44, 12) is out of limitation.
"""


for i in range(100):
    # seed = SeedSequence(54171349637842144159007613790275699200)
    # print("entropy={}".format(seed.entropy))

    seed = SeedSequence()
    print("test #{}, entropy={}".format(i + 1, seed.entropy))

    print("[Random]")
    env = Environment(environment_configuration, seed)
    random_alg = RandomAssignmentAllocation(env)
    random_alg.run()

    # seed = SeedSequence(123)
    print("[Nearest]")
    env = Environment(environment_configuration, seed)
    nearest_alg = NearestAssignmentAllocation(env)
    nearest_alg.run()

    # print("[Greedy]")
    # env = Environment(environment_configuration, seed)
    # greedy_alg = GreedyAssignmentAllocation(env)
    # greedy_alg.run()



    # algorithm = RandomAssignmentAllocation(env)
    # algorithm.run()