from codes.min_cost_unsharedA.env.environment import *
import torch
from math import sqrt
import numpy as np
from torch_geometric.data import Data

class PackagedEnv:
    def __init__(self, env: Environment):
        self.env = env

        self.num_edge_node = env.num_edge_node

        self.node_state_dim = 13
        self.user_state_dim = self.env.num_edge_node + 2

        # 节点的连接关系、边的特征，都是不变的
        self.edge_index = None
        self.edge_attr = None
        self.init_edge_index_attr()

        self.assigned_users = []  # 已经完成关联、服务器分配的用户
        self.current_user_id = 0

        self.cost = 0

        self.debug_flag = False  # True = On, False = Off


    """
        重置环境，并返回原始状态
    """
    def reset(self):
        for node in self.env.edge_node_list:
            node.reset()
        for user in self.env.user_list:
            user.reset()
        self.env.service_b.reset()

        self.assigned_users.clear()
        self.current_user_id = 0

        self.initialize()
        return self.get_state()

    """
        执行action，这里的action表示选出的服务A的位置。R选一个负载最低的。
        返回：(s', r, done, _)
    """
    def step(self, action):
        reward = 0
        done = False

        user = self.env.user_list[self.current_user_id]
        self.assigned_users.append(user)

        # node_a_id = action
        # node_r_id = sorted(self.env.edge_node_list, key=lambda x: (x.num_server / x.capacity + x.price["R"] / 6), reverse=False)[0].node_id

        node_a_id = sorted(self.env.edge_node_list, key=lambda x: (x.num_server / x.capacity + x.price["A"] / 5), reverse=False)[0].node_id
        node_r_id = action

        node_a = self.env.edge_node_list[node_a_id]
        node_r = self.env.edge_node_list[node_r_id]

        # 不满足时延约束，给予惩罚
        if not self.is_tx_tp_satisfied(user, node_a, node_r):
            reward = -50 * (len(self.assigned_users) / self.env.num_user)
            done = True
            self.assigned_users.pop(-1)

        else:

            self.assign_and_initialize(user.service_A, node_a)
            self.assign_and_initialize(user.service_R, node_r)

            # 分配服务器使交互时延降低至 T_limit 以下
            self.allocate_for_delay_limitations(user)

            new_cost = self.env.compute_cost(self.assigned_users)
            delta_cost = new_cost - self.cost
            self.cost = new_cost

            # reward += -sqrt(delta_cost / len(self.assigned_users))
            # reward += 10 * len(self.assigned_users) / self.env.num_user

            reward += 100 / delta_cost

            # reward += -delta_cost

        self.current_user_id += 1
        if self.current_user_id == self.env.num_user:
            done = True

        graph_s, user_s = self.get_state()
        return graph_s, user_s, reward, done, None


    """
        获取当前的状态
        1. 节点的状态
        2. 下一个待分配用户的状态
    """
    def get_state(self):
        node_state_list = []
        for node in self.env.edge_node_list:    # type: EdgeNode
            node_state = list()

            node_state.append(node.service_rate["A"])
            node_state.append(node.service_rate["B"])
            node_state.append(node.service_rate["R"])
            node_state.append(node.price["A"])
            node_state.append(node.price["B"])
            node_state.append(node.price["R"])
            node_state.append(node.extra_price["A"])
            node_state.append(node.extra_price["B"])
            node_state.append(node.extra_price["R"])
            node_state.append(node.capacity)
            node_state.append(node.num_server)

            max_delay_UA = 0
            max_delay_RU = 0
            for u in self.assigned_users:    # type: User
                if u.service_A in node.service_list.values():
                    max_delay_UA = max(max_delay_UA, self.env.tx_user_node[u.user_id][node.node_id])
                if u.service_R in node.service_list.values():
                    max_delay_RU = max(max_delay_RU, self.env.tx_user_node[u.user_id][node.node_id])
            node_state.append(max_delay_UA)
            node_state.append(max_delay_RU)

            node_state = np.array(node_state)

            # 最大最小值归一化
            # node_state = (node_state - np.min(node_state)) / (np.max(node_state) - np.min(node_state))

            node_state_list.append(node_state)

        graph = Data(x=torch.tensor(np.array(node_state_list), dtype=torch.float),
                     edge_index=self.edge_index,
                     edge_attr=self.edge_attr)

        user_state = []
        user = self.env.user_list[self.current_user_id if self.current_user_id != self.env.num_user else self.env.num_user - 1]
        user_state.append(user.arrival_rate)
        user_state.append(self.env.delay_limit * 1000)
        for delay in self.env.tx_user_node[user.user_id]:
            user_state.append(delay)

        user_state = np.array(user_state)

        # 最大最小值归一化
        # user_state = (user_state - np.min(user_state)) / (np.max(user_state) - np.min(user_state))

        return graph, user_state


    """
        初始化
        self.edge_index, self.edge_weight
    """
    def init_edge_index_attr(self):
        self.edge_index = [[], []]
        self.edge_attr = []
        for i in range(self.env.num_edge_node):
            for j in range(self.env.num_edge_node):
                if i == j:
                    continue
                self.edge_index[0].append(i)
                self.edge_index[1].append(j)

        self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)

        for i in range(self.env.num_edge_node):
            for j in range(self.env.num_edge_node):
                if i == j:
                    continue
                # self.edge_attr.append([self.env.tx_node_node[i][j], self.env.t_price_node_node[i][j]])
                self.edge_attr.append([self.env.tx_node_node[i][j]])

        # for i in range(self.env.num_edge_node):
        #     for j in range(self.env.num_edge_node):
        #         if i == j:
        #             continue
        #         self.edge_attr.append(self.env.tx_node_node[i][j])

        self.edge_attr = torch.tensor(self.edge_attr, dtype=torch.float)


    """
        初始化service b的位置：选择到其它节点时延之和最小的节点，当作服务B的关联节点
    """
    def initialize(self):
        service_b = self.env.service_b
        min_delay_sum = 1e15
        target_node = None

        for edge_node in self.env.edge_node_list:  # type: EdgeNode
            delay_sum = 0
            for i in range(self.env.num_edge_node):
                delay_sum += self.env.tx_node_node[edge_node.node_id][i]
            if delay_sum < min_delay_sum:
                min_delay_sum = delay_sum
                target_node = edge_node

        self.assign_and_initialize(service_b, target_node)

        self.cost = self.env.compute_cost(self.assigned_users)

        self.DEBUG("service B: node_id={}, service_rate={}".format(service_b.node_id, service_b.service_rate))

    """
        将service关联到给定的EdgeNode，并分配若干服务器，初始化满足稳态条件
    """
    def assign_and_initialize(self, service: Service, edge_node: EdgeNode):
        num_server = service.get_num_server_for_stability(edge_node.service_rate[service.service_type])
        extra_num_server = self.compute_num_extra_server(num_server, edge_node)

        service.node_id = edge_node.node_id
        service.service_rate = edge_node.service_rate[service.service_type]
        service.price = edge_node.price[service.service_type]
        service.extra_price = edge_node.extra_price[service.service_type]
        service.update_num_server(n=num_server, extra_n=extra_num_server, update_queuing_delay=True)

        edge_node.service_list[(service.user_id, service.service_type)] = service
        edge_node.num_server += num_server
        edge_node.num_extra_server += extra_num_server

    """
        计算需要的额外空间
    """
    def compute_num_extra_server(self, num: int, edge_node: EdgeNode) -> int:
        extra_num_server = 0
        if edge_node.num_server >= edge_node.capacity:
            extra_num_server = num
        else:
            extra_num_server = num - (edge_node.capacity - edge_node.num_server)
            if extra_num_server < 0:
                extra_num_server = 0
        return extra_num_server

    """
        计算与给定用户相关的时延对的 Tx + Tp，查看是否都小于 T_limit
    """
    def is_tx_tp_satisfied(self, user: User, node_a: EdgeNode, node_r: EdgeNode) -> bool:
        user.service_A.node_id = node_a.node_id
        user.service_A.service_rate = node_a.service_rate[user.service_A.service_type]
        user.service_R.node_id = node_r.node_id
        user.service_R.service_rate = node_r.service_rate[user.service_R.service_type]

        flag = True
        for u in self.assigned_users:  # type: User
            tx_tp = self.env.compute_tx_tp(user, u)
            if tx_tp >= self.env.delay_limit:
                flag = False
                break

            tx_tp = self.env.compute_tx_tp(u, user)
            if tx_tp >= self.env.delay_limit:
                flag = False
                break

        user.service_A.reset()
        user.service_R.reset()
        return flag

    """
        分配服务器以满足时延约束，每次选择 reduction / price 最大的
    """
    def allocate_for_delay_limitations(self, cur_user: User):
        user_from, user_to, max_delay = self.env.compute_max_interactive_delay_by_given_user(cur_user, self.assigned_users)
        while max_delay > self.env.delay_limit:
            # 为当前服务链分配服务器，直到其降低到时延约束以下
            services = self.env.get_service_chain(user_from, user_to)
            services.sort(key=lambda x: x.reduction_of_delay_when_add_a_server() / x.price, reverse=True)
            selected_service = services[0]
            self.allocate(selected_service, 1)

            user_from, user_to, max_delay = self.env.compute_max_interactive_delay_by_given_user(cur_user, self.assigned_users)


    """
        为某个服务增加若干服务器
    """
    def allocate(self, service: Service, num: int):
        edge_node = self.env.edge_node_list[service.node_id]  # type:EdgeNode
        extra_num = self.compute_num_extra_server(num, edge_node)

        edge_node.num_server += num
        edge_node.num_extra_server += extra_num
        service.update_num_server(service.num_server + num, service.num_extra_server + extra_num,
                                  update_queuing_delay=True)

    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)