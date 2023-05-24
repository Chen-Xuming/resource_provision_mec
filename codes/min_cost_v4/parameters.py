"""
    Environment 参数
"""

# environment_configuration = {
#     "num_user": 100,
#     "num_edge_node": 50,
#     "minmax_arrival_rate": (8, 16),
#     "minmax_service_rate_A": (15, 20),
#     "minmax_service_rate_B": (150, 200),
#     "minmax_service_rate_R": (60, 80),
#     "minmax_price_A": (1, 3),
#     "minmax_price_B": (15, 20),
#     "minmax_price_R": (3, 6),
#     "minmax_edge_node_capacity": (60, 100),
#     "minmax_transmission_delay": (5, 20),
#     "minmax_transmission_price": (1, 5),
#     "transmission_data_size": (1, 1, 1, 2),
#     "trigger_probability": 0.6,
#     "delay_limit": 0.14   # 秒
# }

# environment_configuration = {
#     "num_user": 100,
#     "num_edge_node": 50,
#     "minmax_arrival_rate": (5, 15),             #
#     "minmax_service_rate_A": (50, 70),          # [14.3, 20] ms
#     "minmax_service_rate_B": (150, 200),        # [5, 6.6] ms
#     "minmax_service_rate_R": (30, 45),          # [22.2, 33.3] ms   ==> tp = [41, 60]
#     "minmax_price_A": (1, 3),
#     "minmax_price_B": (5, 10),
#     "minmax_price_R": (3, 6),
#     "price_times_of_extra_server": 2,           # 超出容量的服务器，价格是原价的若干倍
#     "minmax_edge_node_capacity": (60, 100),
#     "minmax_transmission_delay": (5, 16),       # tx = [20, 64] ms,  tx + tp = [61, 124]
#     "minmax_transmission_price": (2, 10),
#     "transmission_data_size": (1, 1, 1, 5),
#     "trigger_probability": 0.6,
#     "delay_limit": 0.1   # 秒
# }

# environment_configuration = {
#     "num_user": 50,
#     "num_edge_node": 25,
#     "minmax_arrival_rate": (5, 15),             #
#     "minmax_service_rate_A": (50, 70),          # [14.3, 20] ms
#     "minmax_service_rate_B": (150, 200),        # [5, 6.6] ms
#     "minmax_service_rate_R": (30, 45),          # [22.2, 33.3] ms   ==> tp = [41, 60]
#     "minmax_price_A": (1, 3),
#     "minmax_price_B": (5, 10),
#     "minmax_price_R": (3, 6),
#     "price_times_of_extra_server": 2,           # 超出容量的服务器，价格是原价的若干倍
#     "minmax_edge_node_capacity": (10, 20),
#     "minmax_transmission_delay": (5, 10),       # tx = [20, 64] ms,  tx + tp = [61, 124]
#     "minmax_transmission_price": (2, 5),
#     "transmission_data_size": (1, 1, 1, 5),
#     "trigger_probability": 0.6,
#     "delay_limit": 0.1   # 秒
# }

environment_configuration = {
    "num_user": 50,
    "num_edge_node": 40,
    "minmax_arrival_rate": (5, 15),             #
    "minmax_service_rate_A": (50, 70),          # [14.3, 20] ms
    "minmax_service_rate_B": (150, 200),        # [5, 6.6] ms
    "minmax_service_rate_R": (30, 45),          # [22.2, 33.3] ms   ==> tp = [41, 60]
    "minmax_price_A": (2, 5),
    "minmax_price_B": (5, 10),
    "minmax_price_R": (4, 8),
    "price_times_of_extra_server": 2,           # 超出容量的服务器，价格是原价的若干倍
    "minmax_edge_node_capacity": (20, 40),
    "minmax_transmission_delay": (5, 16),       # tx = [20, 64] ms,  tx + tp = [61, 124]
    "minmax_transmission_price": (2, 5),
    "transmission_data_size": (1, 1, 1, 5),
    "trigger_probability": 0.6,
    "delay_limit": 0.100   # 秒
}

def show_env_config():
    print("-------------- Configurations -------------------")
    for k, v in environment_configuration.items():
        print("\"{}\": {}".format(k, v))

    print("-------------------------------------------------")

