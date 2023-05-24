from numpy.random import default_rng, SeedSequence
from codes.min_cost_v4.env.edge_node import EdgeNode
from codes.min_cost_v4.env.user import User
from codes.min_cost_v4.env.service import Service
import numpy as np


class Environment:
    def __init__(self, parameters: dict, seed_sequence: SeedSequence):
        self.config = parameters

        # 注意：此处的随机种子仅仅用于生成底层网络相关的数据，对于用户相关的数据，用其他随机种子生成，因为训练过程底层网络配置不变、用户变化。
        self.seed_sequence = seed_sequence     # 随机种子
        self.rng = default_rng(seed_sequence)  # 随机数生成器

        self.num_user = None    # 用户数量待定，可变化
        self.num_edge_node = parameters["num_edge_node"]  # 边缘节点数量

        # 用户到达率取值范围
        self.minmax_arrival_rate = parameters["minmax_arrival_rate"]

        # 各类服务的服务率取值范围
        self.minmax_service_rate_A = parameters["minmax_service_rate_A"]
        self.minmax_service_rate_B = parameters["minmax_service_rate_B"]
        self.minmax_service_rate_R = parameters["minmax_service_rate_R"]

        # 各类型服务器的单价取值范围
        # 每个 EdgeNode 拥有一组 (price_a, price_b, price_r)
        self.minmax_price_A = parameters["minmax_price_A"]
        self.minmax_price_B = parameters["minmax_price_B"]
        self.minmax_price_R = parameters["minmax_price_R"]

        # 超出容量的服务器，价格是原价的若干倍
        self.price_times_of_extra_server = parameters["price_times_of_extra_server"]

        # 每个 EdgeNode 的 capacity 的取值范围
        self.minmax_edge_node_capacity = parameters["minmax_edge_node_capacity"]

        # 节点间的传输时延取值范围（包括 user <--> edge node）
        self.minmax_transmission_delay = parameters["minmax_transmission_delay"]
        """
            (1) user <--> edge node 传输时延矩阵   (num_user, num_edge_node)
            (2) edge node <--> edge node 传输时延矩阵   (num_edge_node, num_edge_node)

            注意：1. tx_node_node[i][i] = 0
                 2. tx_node_node[i][j] = tx_node_node[j][i],  即同一条链路传输时延是一样的
        """
        # self.tx_user_node = self.rng.integers(self.minmax_transmission_delay[0],
        #                                       self.minmax_transmission_delay[1] + 1,
        #                                       (self.num_user, self.num_edge_node))
        self.tx_user_node = None    # 初始化用户的时候再赋值

        self.tx_node_node = np.zeros((self.num_edge_node, self.num_edge_node))
        for i in range(self.num_edge_node):
            for j in range(i + 1, self.num_edge_node):
                self.tx_node_node[i][j] = self.rng.integers(self.minmax_transmission_delay[0],
                                                            self.minmax_transmission_delay[1] + 1)
                self.tx_node_node[j][i] = self.tx_node_node[i][j]

        self._trigger_probability = parameters["trigger_probability"]  # 触发概率

        self.delay_limit = parameters["delay_limit"]

        self.user_list = []       # 用户列表
        self.edge_node_list = []  # 边缘节点列表

        self.total_arrival_rate = 0
        self.service_b = Service(service_type="B", node_id=None, user_id=None)

        # self.initialize_users()
        # self.initialize_arrival_rate()
        self.initialize_edge_nodes()

    """
        根据给随机种子初始化用户及其服务
        
        注意：此函数是在Environment对象生成之后，由用户手动调用
    """
    def set_users_and_services_by_given_seed(self, user_seed, num_user):
        # 用给定的随机种子初始化随机数生成器
        user_seed_sequence = SeedSequence(user_seed)
        user_rng = default_rng(user_seed_sequence)

        # 清除原来的用户集合
        self.user_list.clear()

        # 更改系统用户数
        self.num_user = num_user

        # 重置服务B
        self.service_b.reset()

        # 重置系统到达率之和
        self.total_arrival_rate = 0

        # 根据给定的用户数量，创建新的 user-node 传输时延矩阵
        self.tx_user_node = None
        self.tx_user_node = user_rng.integers(self.minmax_transmission_delay[0],
                                              self.minmax_transmission_delay[1] + 1,
                                              (self.num_user, self.num_edge_node))

        # 生成 num_user 个用户及其服务。
        for user_id in range(self.num_user):
            user = User(user_id=user_id)
            user.arrival_rate = user_rng.integers(self.minmax_arrival_rate[0], self.minmax_arrival_rate[1] + 1)

            service_a = Service(service_type="A", user_id=user_id)
            service_r = Service(service_type="R", user_id=user_id)
            user.service_A = service_a
            user.service_R = service_r
            user.service_B = self.service_b

            self.user_list.append(user)

        # 初始化各个服务的到达率
        for user in self.user_list:     # type: User
            self.total_arrival_rate += user.arrival_rate
        self.service_b.arrival_rate = self.total_arrival_rate
        for user in self.user_list:     # type: User
            user.service_A.arrival_rate = user.arrival_rate
            user.service_R.arrival_rate = int(self.total_arrival_rate * self._trigger_probability)

    # """
    #     初始化用户，同时为他们初始化各自的服务A/B/R。
    #     注意：服务B只有一个。
    # """
    # def initialize_users(self):
    #     for user_id in range(self.num_user):
    #         user = User(user_id=user_id)
    #         user.arrival_rate = self.rng.integers(self.minmax_arrival_rate[0], self.minmax_arrival_rate[1] + 1)
    #
    #         service_a = Service(service_type="A", user_id=user_id)
    #         service_r = Service(service_type="R", user_id=user_id)
    #         user.service_A = service_a
    #         user.service_R = service_r
    #         user.service_B = self.service_b
    #
    #         self.user_list.append(user)

    # """
    #     初始化各个服务的到达率
    # """
    # def initialize_arrival_rate(self):
    #     for user in self.user_list:  # type: User
    #         self.total_arrival_rate += user.arrival_rate
    #
    #     self.service_b.arrival_rate = self.total_arrival_rate
    #
    #     for user in self.user_list:  # type: User
    #         user.service_A.arrival_rate = user.arrival_rate
    #         user.service_R.arrival_rate = int(self.total_arrival_rate * self._trigger_probability)

    """
        初始化边缘节点：容量、单价、服务率
    """
    def initialize_edge_nodes(self):
        for i in range(self.num_edge_node):
            edge_node = EdgeNode(i)
            edge_node.capacity = self.rng.integers(self.minmax_edge_node_capacity[0],
                                                   self.minmax_edge_node_capacity[1] + 1)

            # 服务单价
            edge_node.price["A"] = self.rng.integers(self.minmax_price_A[0], self.minmax_price_A[1] + 1)
            edge_node.price["B"] = self.rng.integers(self.minmax_price_B[0], self.minmax_price_B[1] + 1)
            edge_node.price["R"] = self.rng.integers(self.minmax_price_R[0], self.minmax_price_R[1] + 1)

            # 超出capacity时的服务单价
            edge_node.extra_price["A"] = edge_node.price["A"] * self.price_times_of_extra_server
            edge_node.extra_price["B"] = edge_node.price["B"] * self.price_times_of_extra_server
            edge_node.extra_price["R"] = edge_node.price["R"] * self.price_times_of_extra_server

            # 服务率
            edge_node.service_rate["A"] = self.rng.integers(self.minmax_service_rate_A[0],
                                                            self.minmax_service_rate_A[1] + 1)
            edge_node.service_rate["B"] = self.rng.integers(self.minmax_service_rate_B[0],
                                                            self.minmax_service_rate_B[1] + 1)
            edge_node.service_rate["R"] = self.rng.integers(self.minmax_service_rate_R[0],
                                                            self.minmax_service_rate_R[1] + 1)
            self.edge_node_list.append(edge_node)

    """
        计算开销(已关联且分配服务器的用户)
        服务器分配开销 = 所有用户的每个service分配服务器的开销之和，注意：服务B只需要计算一次。
        传输开销包含交互路径上的四段。
    """
    def compute_cost(self):
        allocation_cost = 0

        # 服务器分配开销
        for node in self.edge_node_list:  # type: EdgeNode
            for service in node.service_list.values():  # type: Service
                out_capacity = service.num_extra_server
                in_capacity = service.num_server - out_capacity
                allocation_cost += in_capacity * service.price
                allocation_cost += out_capacity * service.extra_price

        total_cost = allocation_cost
        return total_cost

    """
        计算两个用户之间的交互时延
    """
    def compute_interactive_delay(self, user_from: User, user_to: User) -> float:
        # 传输时延(ms)
        transmission_delay = self.tx_user_node[user_from.user_id][user_from.service_A.node_id] + \
                             self.tx_node_node[user_from.service_A.node_id][self.service_b.node_id] + \
                             self.tx_node_node[self.service_b.node_id][user_to.service_R.node_id] + \
                             self.tx_user_node[user_to.user_id][user_to.service_R.node_id]
        transmission_delay = transmission_delay / 1000  # 将单位换算为秒

        # 排队时延(s)
        queuing_delay = user_from.service_A.queuing_delay + self.service_b.queuing_delay + user_to.service_R.queuing_delay

        # 处理时延(s)
        processing_delay = 1 / user_from.service_A.service_rate + \
                           1 / self.service_b.service_rate + \
                           1 / user_to.service_R.service_rate

        interactive_delay = transmission_delay + queuing_delay + processing_delay
        return interactive_delay

    """
        计算 Tx + Tp
    """
    def compute_tx_tp(self, user_from: User, user_to: User) -> float:
        # 传输时延(ms)
        transmission_delay = self.tx_user_node[user_from.user_id][user_from.service_A.node_id] + \
                             self.tx_node_node[user_from.service_A.node_id][self.service_b.node_id] + \
                             self.tx_node_node[self.service_b.node_id][user_to.service_R.node_id] + \
                             self.tx_user_node[user_to.user_id][user_to.service_R.node_id]
        transmission_delay = transmission_delay / 1000  # 将单位换算为秒

        # 处理时延(s)
        processing_delay = 1 / user_from.service_A.service_rate + \
                           1 / self.service_b.service_rate + \
                           1 / user_to.service_R.service_rate

        return transmission_delay + processing_delay

    """
        计算用户user与所有用户（包括自己）的最大交互时延
        1. user --> 其它用户
        2. 其他用户 --> user
        返回最大时延以及相应的用户对
    """
    def compute_max_interactive_delay_by_given_user(self, cur_user: User, assigned_user_list: list) -> (User, User, float):
        max_delay = -1
        user_pair = (-1, -1)

        for other_user in assigned_user_list:  # type: User
            delay = self.compute_interactive_delay(cur_user, other_user)
            if delay > max_delay:
                max_delay = delay
                user_pair = (cur_user, other_user)

            delay = self.compute_interactive_delay(other_user, cur_user)
            if delay > max_delay:
                max_delay = delay
                user_pair = (other_user, cur_user)

        return user_pair[0], user_pair[1], max_delay

    def compute_max_interactive_delay(self, assigned_user_list: list) -> (User, User, float):
        max_delay = -1
        user_pair = (-1, -1)

        for user_from in assigned_user_list:  # type: User
            for user_to in assigned_user_list:  # type: User
                delay = self.compute_interactive_delay(user_from, user_to)
                if delay > max_delay:
                    max_delay = delay
                    user_pair = (user_from, user_to)

        return user_pair[0], user_pair[1], max_delay

    """
        获取一条服务链
    """
    def get_service_chain(self, user_from: User, user_to: User) -> list:
        return [user_from.service_A, self.service_b, user_to.service_R]

