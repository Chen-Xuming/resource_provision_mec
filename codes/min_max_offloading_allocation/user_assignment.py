"""
    确定用户 - 服务关联关系。

    Environment 统一调用 assign_users_to_services()，再根据实参 “assignment_algorithm” 调用具体的算法。
"""

# from environment import Env

def assign(environment, assignment_algorithm: str):
    assignment_functions = {
        "random": random_assignment,
        "nearest": nearest_assignment,
        "min_tx_tp": min_tx_tp_assignment
    }
    assignment_functions[assignment_algorithm](environment)

"""
    随机关联
"""
def random_assignment(env):
    for user in env._users:
        user._service_a = env._rng.integers(0, env._num_service_a)
        user._service_r = env._rng.integers(0, env._num_service_r)

        # 二维矩阵保存关联关系
        env.user_service_a[user._id][user._service_a] = 1
        env.user_service_r[user._id][user._service_r] = 1

"""
    就近关联
"""
def nearest_assignment(env):
    for user in env._users:
        # 找出与user传输时延最小的服务a
        min_tx = 10e10
        service_id = None
        for sid, tx in enumerate(env._tx_u_a[user._id]):
            if tx < min_tx:
                min_tx = tx
                service_id = sid
        user._service_a = service_id
        env.user_service_a[user._id][user._service_a] = 1

    for user in env._users:
        min_tx = 10e10
        service_id = None
        for sid in range(env._num_service_r):
            tx = env._tx_r_u[sid][user._id]
            if tx < min_tx:
                min_tx = tx
                service_id = sid
        user._service_r = service_id
        env.user_service_r[user._id][user._service_r] = 1

"""
    只考虑传输时延和排队时延的DP关联算法
"""
def min_tx_tp_assignment(env):
    services_info = env._services_config

    # ------ user - a - q -----
    for user in env._users:
        min_delay_service = None
        min_tx = 10e10
        for sid in range(env._num_service_a):
            tx_u_a = env._tx_u_a[user._id][sid]
            tp_a = 1.0 / services_info["service_A"][sid][0]
            tx_a_q = env._tx_a_q[sid]
            tp_q = 1.0 / services_info["service_Q"][0]
            delay = tx_u_a + tp_a + tx_a_q + tp_q
            if delay < min_tx:
                min_tx = delay
                min_delay_service = sid
        user._service_a = min_delay_service
        env.user_service_a[user._id][user._service_a] = 1

    # ------ q - r - user -----
    for user in env._users:
        min_delay_service = None
        min_tx = 10e10
        for sid in range(env._num_service_r):
            tx_r_u = env._tx_r_u[sid][user._id]
            tp_r = 1.0 / services_info["service_R"][sid][0]
            tx_q_r = env._tx_q_r[sid]
            tp_q = 1.0 / services_info["service_Q"][0]
            delay = tp_q + tx_q_r + tp_r + tx_r_u
            if delay < min_tx:
                min_tx = delay
                min_delay_service = sid
        user._service_r = min_delay_service
        env.user_service_r[user._id][user._service_r] = 1