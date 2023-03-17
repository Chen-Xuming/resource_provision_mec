from environment import *
from algorithm import *
from numpy.random import default_rng, SeedSequence

"""
    Environment配置
"""
num_instance = 1
num_service = 10
num_group_per_instance = 10
env_config = {
    "num_instance": num_instance,
    "num_service_a": num_service,
    "num_service_r": num_service,
    "budget_addition": 20,
    "num_group_per_instance": num_group_per_instance,
    "num_user_per_group": 10,
    "min_arrival_rate": 10,
    "max_arrival_rate": 15,
    "min_max_service_rate_a": (100, 200),
    "min_max_service_rate_q": (200, 500),
    "min_max_service_rate_r": (40, 50),
    "min_price": 1,
    "max_price": 5,
    "trigger_probability": 0.2,
    "tx_ua_min": 4,
    "tx_ua_max": 6,
    "tx_aq_min": 2,
    "tx_aq_max": 4,
    "tx_qr_min": 2,
    "tx_qr_max": 4,
    "tx_ru_min": 4,
    "tx_ru_max": 6
}

seed_sequence = SeedSequence()
rng = default_rng(seed_sequence)

env_config["budget_addition"] = 150
env_for_copy = Env(env_config, rng, seed_sequence)

gcn_alg = min_max_gcn(env=env_for_copy)
gcn_alg.set_num_server()