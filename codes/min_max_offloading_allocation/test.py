from numpy.random import default_rng, SeedSequence
from environment import *

from allocation import *

"""
    Environment配置
"""
num_instance = 1
num_service = 10

env_config = {
    "num_instance": num_instance,
    "num_service_a": num_service,
    "num_service_r": num_service,
    "budget_addition": 200,
    "num_group_per_instance": 10,
    "num_user_per_group": 10,
    "min_arrival_rate": 10,         # 个 / 秒
    "max_arrival_rate": 15,
    "min_service_rate_a": 100,      # 个 / 秒
    "max_service_rate_a": 200,
    "min_service_rate_q": 200,
    "max_service_rate_q": 500,
    "min_service_rate_r": 40,
    "max_service_rate_r": 50,
    "min_price": 1,
    "max_price": 5,
    "trigger_probability": 0.2,
    "tx_ua_min": 5,
    "tx_ua_max": 20,
    "tx_aq_min": 5,
    "tx_aq_max": 15,
    "tx_qr_min": 5,
    "tx_ru_min": 15,
    "tx_qr_max": 5,
    "tx_ru_max": 20,
    "assignment_algorithm": "random"
}

def set_assignment_algorithm(alg: str):
    env_config["assignment_algorithm"] = alg

set_assignment_algorithm("random")
# set_assignment_algorithm("nearest")
# set_assignment_algorithm("dp")


assign_algorithms = ["random", "nearest", "min_tx_tp"]
for seed in range(20, 30):
    print("------------------------------------- seed = {} ------------------------------------".format(seed))
    for asg_alg in assign_algorithms:
        set_assignment_algorithm(alg=asg_alg)
        seed_sequence = SeedSequence(seed)
        rng = default_rng(seed_sequence)
        env = Env(env_config, rng, seed_sequence)

        allocation_algorithm = min_max_greedy(env)
        allocation_algorithm.set_num_server()
