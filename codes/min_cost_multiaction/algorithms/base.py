from codes.min_cost_multiaction.env.edge_node import EdgeNode
from codes.min_cost_multiaction.env.service import Service
from codes.min_cost_multiaction.env.user import User
from codes.min_cost_multiaction.env.environment import Environment


from time import time


class BaseAlgorithm:
    def __init__(self, env: Environment, *args, **kwargs):
        self.algorithm_name = None

        self.env: Environment = env

        self.start_time = None
        self.end_time = None
        self.running_time = None    # 算法运行时间

        self.final_cost = None      # 最终开销

        self.avg_delay = None       # 平均交互时延

        self.results = dict()       # 结果记录（运行时间，cost等）

    def get_running_time(self):
        self.end_time = time()
        self.running_time = (self.end_time - self.start_time) * 1000    # ms

    """
        开始运行，具体的逻辑由各算法类实现
    """
    def run(self):
        pass

    """
        检查结果是否正确
        0. 检查每个服务的排队时延是否跟num_ser对应的排队时延一致
        1. 检查是否所有用户都关联了EdgeNode，service的价格和服务率是否一致
        2. 检查是否所有时延满足约束
        3. 检查服务器数量是否正确
    """
    def check_result(self):
        error_range = 1e-10
        for user in self.env.user_list:     # type: User
            delay = user.service_A.compute_queuing_delay(user.service_A.num_server)
            qd = user.service_A.queuing_delay
            assert abs(delay - qd) < error_range, "user{}.service_A: Queuing delay is incorrect.".format(user.user_id)

            delay = user.service_B.compute_queuing_delay(user.service_B.num_server)
            qd = user.service_B.queuing_delay
            assert abs(delay - qd) < error_range, "user{}.service_B: Queuing delay is incorrect.".format(user.user_id)

            delay = user.service_R.compute_queuing_delay(user.service_R.num_server)
            qd = user.service_R.queuing_delay
            assert abs(delay - qd) < error_range, "user{}.service_R: Queuing delay is incorrect.".format(user.user_id)

        total_delay = 0
        for user_from in self.env.user_list:     # type: User
            s_a, s_b, s_r = user_from.service_A, user_from.service_B, user_from.service_R

            assert s_a.node_id is not None, "Service_A of user_{} hasn't been attached to any EdgeNode.".format(user_from.user_id)
            assert s_b.node_id is not None, "Service_B of user_{} hasn't been attached to any EdgeNode.".format(user_from.user_id)
            assert s_r.node_id is not None, "Service_R of user_{} hasn't been attached to any EdgeNode.".format(user_from.user_id)

            assert s_a.service_rate == self.env.edge_node_list[s_a.node_id].service_rate[s_a.service_type], "Service rate error"
            assert s_b.service_rate == self.env.edge_node_list[s_b.node_id].service_rate[s_b.service_type], "Service rate error"
            assert s_r.service_rate == self.env.edge_node_list[s_r.node_id].service_rate[s_r.service_type], "Service rate error"

            assert s_a.price == self.env.edge_node_list[s_a.node_id].price[s_a.service_type], "Price error"
            assert s_b.price == self.env.edge_node_list[s_b.node_id].price[s_b.service_type], "Price error"
            assert s_r.price == self.env.edge_node_list[s_r.node_id].price[s_r.service_type], "Price error"

            for user_to in self.env.user_list:   # type: User
                delay = self.env.compute_interactive_delay(user_from, user_to)
                assert delay <= self.env.delay_limit, "Interactive delay of users ({}, {}) is out of limitation.".format(user_from.user_id, user_to.user_id)

                total_delay += delay

        self.avg_delay = total_delay / (self.env.num_user**2)


        # ---------------- 检查服务器数量是否正确 -------------------------
        for node in self.env.edge_node_list:        # type: EdgeNode
            num_server = 0
            extra_num_server = 0
            for s in node.service_list.values():        # type: Service
                num_server += s.num_server
                extra_num_server += s.num_extra_server
            assert num_server == node.num_server
            assert extra_num_server == node.num_extra_server



    """
        计算最终的开销
    """
    def compute_final_cost(self):
        self.check_result()

        self.final_cost = self.env.compute_cost(self.env.user_list)
        print("[{}]:  [final_cost]: {},  [average_delay]: {},  [running_time]: {}".format(self.algorithm_name,
                                                                                          self.final_cost,
                                                                                          self.avg_delay,
                                                                                          self.running_time))

    """
        获取最终的统计信息
    """
    def get_results(self):
        self.results["cost"] = self.final_cost
        self.results["running_time"] = self.running_time
        self.results["avg_delay"] = self.avg_delay
        return self.results
