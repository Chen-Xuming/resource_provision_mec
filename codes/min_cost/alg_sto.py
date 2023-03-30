from alg_base import *
import random

class RandomAssignmentAllocation(BaseAlgorithm):
    def __init__(self, env, *args, **kwargs):
        BaseAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "min_cost_random" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        random.seed(env.seed_sequence.entropy)

        self.assigned_users = []    # 已经完成关联、服务器分配的用户

        self.debug_flag = False  # True = On, False = Off

    def run2(self):
        self.start_time = time()

        # --------------- 为同步服务器B关联一个EdgeNode ---------------
        service_b = self.env.service_b
        node_id = self.env.rng.integers(0, self.env.num_edge_node)
        node_for_b = self.env.edge_node_list[node_id]
        self.assign_and_initialize(service_b, node_for_b)
        self.DEBUG("service B: node_id={}, service_rate={}".format(service_b.node_id, service_b.service_rate))

        # --------------- 为每个用户随机选择两个关联节点并分配服务器，要求满足时延约束 ----------------
        for user in self.env.user_list:     # type: User
            self.assigned_users.append(user)

            success = False
            candidate_pairs = []  # 待尝试的组合，有 num_edge_node ** 2 种
            for i in range(self.env.num_edge_node):
                for j in range(self.env.num_edge_node):
                    candidate_pairs.append((i, j))

            while not success:
                # ----------- 随机抽取一种未尝试过的组合 -------------
                if len(candidate_pairs) == 0:
                    break
                pair = random.sample(candidate_pairs, 1)[0]
                candidate_pairs.remove(pair)

                node_a = self.env.edge_node_list[pair[0]]   # type: EdgeNode
                node_r = self.env.edge_node_list[pair[1]]   # type: EdgeNode

                # 首先要满足 Tx + Tp < T_limit，否则无论如何增加服务器，都无济于事
                if not self.is_tx_tp_satisfied(user, node_a, node_r):
                    continue

                # 如果node_a还没有服务A，则创建一个，并进行初始化；
                # 如果已有，则进行更新
                if (None, "A") not in node_a.service_list:
                    service_A = Service("A", node_a.node_id, user_id=None)
                    service_A.arrival_rate = user.arrival_rate
                    user.service_A = service_A
                    self.assign_and_initialize(user.service_A, node_a)
                else:
                    service_A = node_a.service_list[(None, "A")]
                    self.attach_user_to_serviceA(user, service_A)

                self.assign_and_initialize(user.service_R, node_r)

                # 分配服务器使交互时延降低至 T_limit 以下
                self.allocate_for_delay_limitations(user)
                success = True
                break

            assert success, "Time limitation not satisfied, current user: {}".format(user.user_id)

            self.DEBUG("[assign] user:{} to ({}, {}, {})".format(user.user_id, user.service_A.node_id, user.service_B.node_id,
                                                                 user.service_R.node_id))

        self.get_running_time()
        self.compute_final_cost()
        self.DEBUG("cost = {}".format(self.final_cost))
        self.DEBUG("running_time = {}".format(self.running_time))


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
        将user 加入到 service_A
    """
    def attach_user_to_serviceA(self, user: User, service_a: Service):
        edge_node = self.env.edge_node_list[service_a.node_id]  # type: EdgeNode

        origin_num_server = service_a.num_server
        origin_num_extra = service_a.num_extra_server

        user.service_A = service_a
        service_a.arrival_rate += user.arrival_rate

        num_server_for_stab = service_a.get_num_server_for_stability(edge_node.service_rate[service_a.service_type])

        # 由于新用户的到来，可能需要增加服务器以满足稳态条件
        if num_server_for_stab > origin_num_server:
            edge_node.num_server -= origin_num_server
            edge_node.num_extra_server -= origin_num_extra

            extra_num_server = self.compute_num_extra_server(num_server_for_stab, edge_node)

            edge_node.num_server += num_server_for_stab
            edge_node.num_extra_server += extra_num_server

            service_a.update_num_server(num_server_for_stab, extra_num_server, True)
        else:
            service_a.update_num_server(origin_num_server, origin_num_extra, True)


    """
        计算与给定用户相关的时延对的 Tx + Tp，查看是否都小于 T_limit
    """
    def is_tx_tp_satisfied(self, user: User, node_a: EdgeNode, node_r: EdgeNode) -> bool:
        temp_sa = Service("A", node_id=node_a.node_id, user_id=None)    # 虚假的serviceA，为了传递node_id信息

        user.service_A = temp_sa
        user.service_A.service_rate = node_a.service_rate[user.service_A.service_type]
        user.service_R.node_id = node_r.node_id
        user.service_R.service_rate = node_r.service_rate[user.service_R.service_type]

        flag = True
        for u in self.assigned_users:    # type: User
            tx_tp = self.env.compute_tx_tp(user, u)
            if tx_tp >= self.env.delay_limit:
                flag = False
                break

            tx_tp = self.env.compute_tx_tp(u, user)
            if tx_tp >= self.env.delay_limit:
                flag = False
                break

        user.service_A = None
        del temp_sa
        user.service_R.reset()
        return flag

    """
        分配服务器以满足时延约束
    """
    def allocate_for_delay_limitations(self, cur_user: User):
        share_serviceA_users = []
        for user in self.assigned_users:
            if user.service_A.node_id == cur_user.service_A.node_id:
                share_serviceA_users.append(user)

        # user_from, user_to, max_delay = self.env.compute_max_interactive_delay(self.assigned_users)
        user_from, user_to, max_delay = self.get_max_delay_about_by_user(cur_user, involved_users=share_serviceA_users)
        while max_delay > self.env.delay_limit:

            # 为当前服务链分配服务器，直到其降低到时延约束以下
            cur_delay = max_delay
            services = self.env.get_service_chain(user_from, user_to)
            while cur_delay > self.env.delay_limit:
                selected_service = random.sample(services, 1)[0]
                self.allocate(selected_service, 1)
                cur_delay = self.env.compute_interactive_delay(user_from, user_to)

            user_from, user_to, max_delay = self.get_max_delay_about_by_user(cur_user, involved_users=share_serviceA_users)

    """
        获取跟cur_user相关的最大时延
        1. cur_user --> 其它已分配的用户
        2. 其它已分配的用户 --> cur_user
        3. 跟cur_user共享服务A的用户 --> cur_user
    """
    def get_max_delay_about_by_user(self, cur_user: User, involved_users: list):
        user_from, user_to, max_delay = self.env.compute_max_interactive_delay_by_given_user(cur_user, self.assigned_users)

        for user in involved_users:     # type: User
            delay = self.env.compute_interactive_delay(user, cur_user)
            if delay > max_delay:
                user_from = user
                user_to = cur_user
                max_delay = delay

        return user_from, user_to, max_delay


    """
        为某个服务增加若干服务器
    """
    def allocate(self, service: Service, num: int):
        edge_node = self.env.edge_node_list[service.node_id]        # type:EdgeNode
        extra_num = self.compute_num_extra_server(num, edge_node)

        edge_node.num_server += num
        edge_node.num_extra_server += extra_num
        service.update_num_server(service.num_server + num, service.num_extra_server + extra_num, update_queuing_delay=True)


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


    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)
