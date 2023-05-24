from codes.min_cost_v2.algorithms.base import *

class NearestAssignmentAllocation(BaseAlgorithm):
    def __init__(self, env, *args, **kwargs):
        BaseAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "min_cost_nearest" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        self.assigned_users = []  # 已经完成关联、服务器分配的用户

        self.debug_flag = False  # True = On, False = Off

    def run(self):
        self.start_time = time()

        # 为服务B选择EdgeNode，并初始化
        self.initialize_service_B()

        # 为每个用户决定服务A/R的位置，并分配服务器
        self.deploy_for_users()

        self.get_running_time()
        self.compute_final_cost()
        self.DEBUG("cost = {}".format(self.final_cost))
        self.DEBUG("running_time = {}".format(self.running_time))

    """
        选择到其它节点时延之和最小的节点，当作服务B的关联节点
    """
    def initialize_service_B(self):
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
        self.DEBUG("service B: node_id={}, service_rate={}".format(service_b.node_id, service_b.service_rate))


    """
        对于每个用户，选择一组最近的节点.
        先将节点按照距离（时延）排序，然后有 N + N-1 + N-2 + ... + 1 = N(N + 1) / 2 种组合.
        然后，按照类似Random算法的方法进行处理.
    """
    def deploy_for_users(self):
        for user in self.env.user_list:  # type: User
            self.assigned_users.append(user)
            success = False

            node_list = self.get_nearest_edge_node_list(user)  # (node_id, 距离)
            candidate_pairs = []        # list: node_id pairs
            for i in range(len(node_list)):
                for j in range(i, len(node_list)):
                    candidate_pairs.append((node_list[i][0], node_list[j][0]))

            while not success:
                if len(candidate_pairs) == 0:
                    break
                pair = candidate_pairs[0]
                candidate_pairs.pop(0)

                node_a = self.env.edge_node_list[pair[0]]
                node_r = self.env.edge_node_list[pair[1]]

                # 首先要满足 Tx + Tp < T_limit，否则无论如何增加服务器，都无济于事
                if not self.is_tx_tp_satisfied(user, node_a, node_r):
                    continue

                self.assign_and_initialize(user.service_A, node_a)
                self.assign_and_initialize(user.service_R, node_r)

                # 分配服务器使交互时延降低至 T_limit 以下
                self.allocate_for_delay_limitations(user)
                success = True
                break

            assert success, "Time limitation not satisfied, current user: {}".format(user.user_id)
            self.DEBUG("[assign] user:{} to ({}, {}, {})".format(user.user_id, user.service_A.node_id,
                                                                 user.service_B.node_id, user.service_R.node_id))

    """
        获取距离用户 user 最近的节点列表，按照距离排升序
    """
    def get_nearest_edge_node_list(self, user: User) -> list:
        node_list = []
        for node in self.env.edge_node_list:  # type: EdgeNode
            node_list.append((node.node_id, self.env.tx_user_node[user.user_id][node.node_id]))  # (node_id, 距离)

        node_list.sort(key=lambda x: x[1], reverse=False)  # 按照距离排升序

        return node_list

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