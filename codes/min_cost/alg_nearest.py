from alg_base import *

class NearestAssignmentAllocation(BaseAlgorithm):
    def __init__(self, env, *args, **kwargs):
        BaseAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "min_cost_nearest" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        self.assigned_users = []  # 已经完成关联、服务器分配的用户

        self.operation_records = []  # 按时间序记录assign/allocation的操作，用于undo

        self.debug_flag = False  # True = On, False = Off

    def run(self):
        self.start_time = time()

        # 为服务B选择EdgeNode，并初始化
        assert self.initialize_service_B(), "Initialize service B failed."

        # 为每个用户决定服务A/R的位置，并分配服务器
        self.deploy_for_users()

        self.get_running_time()
        self.compute_final_cost()
        self.DEBUG("cost = {}".format(self.final_cost))
        self.DEBUG("running_time = {}".format(self.running_time))

    """
        选择到其它节点时延之和最小的节点，当作服务B的关联节点(要求初始化容量足够)
    """
    def initialize_service_B(self) -> bool:
        service_b = self.env.service_b

        min_delay_sum = 1e15
        target_node = None
        for edge_node in self.env.edge_node_list:  # type: EdgeNode
            if self.is_capacity_enough(service_b, edge_node, None):
                delay_sum = 0
                for i in range(self.env.num_edge_node):
                    delay_sum += self.env.tx_node_node[edge_node.node_id][i]
                if delay_sum < min_delay_sum:
                    min_delay_sum = delay_sum
                    target_node = edge_node

        if target_node is not None:
            self.assign_and_initialize(service_b, target_node, should_record=False, num=None)
            self.DEBUG("service B: node_id={}, service_rate={}".format(service_b.node_id, service_b.service_rate))
            return True

        return False

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

                # 1. 如果容量能够满足稳态约束，则进行初始化
                # 2. 服务器分配，若无法满足时延约束，撤销已进行操作
                if self.is_capacity_enough(user.service_A, node_a) and self.is_capacity_enough(user.service_R, node_r):
                    # step 1
                    # 注意：如果node_a和node_r是同一个，那么分配完之后service_A之后，还要检查capacity
                    self.assign_and_initialize(user.service_A, node_a, should_record=True)
                    if not self.is_capacity_enough(user.service_R, node_r):
                        self.undo_operations()
                        continue
                    else:
                        self.assign_and_initialize(user.service_R, node_r, should_record=True)

                    # step 2
                    if not self.allocate_for_delay_limitations(user):
                        self.undo_operations()
                    else:
                        success = True
                        self.operation_records.clear()  # 因为无需undo，所以可以清空本轮的记录

            assert success, "No any capacity for user {}.".format(user.user_id)
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
    def assign_and_initialize(self, service: Service, edge_node: EdgeNode, should_record=False, num=None):
        num_server = num if num is not None \
            else service.get_num_server_for_stability(edge_node.service_rate[service.service_type])

        service.node_id = edge_node.node_id
        service.service_rate = edge_node.service_rate[service.service_type]
        service.price = edge_node.price[service.service_type]
        service.update_num_server(num_server, update_queuing_delay=True)

        edge_node.service_list[(service.user_id, service.service_type)] = service
        edge_node.num_server += num_server
        assert edge_node.num_server <= edge_node.capacity, "EdgeNode {}: Out of capacity limitation.".format(
            edge_node.node_id)

        if should_record:
            self.record("assignment", service=service, edge_node=edge_node)
            self.record("allocation_for_init", service=service, edge_node=edge_node, num=num_server)

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
        检查一个EdgeNode的剩余容量是否大于给定数量，needed=None默认为service稳态条件所需的数量
    """
    def is_capacity_enough(self, service: Service, edge_node: EdgeNode, server_needed=None):
        if server_needed is None:
            server_needed = service.get_num_server_for_stability(edge_node.service_rate[service.service_type])
        capacity_left = edge_node.capacity - edge_node.num_server
        return server_needed <= capacity_left

    """
        分配服务器以满足时延约束，每次选择 reduction 最大的

    """
    def allocate_for_delay_limitations(self, cur_user: User) -> bool:
        success = True  # 能否将所有时延降低到时延约束以下

        user_from, user_to, max_delay = self.env.compute_max_interactive_delay_by_given_user(cur_user, self.assigned_users)
        while max_delay > self.env.delay_limit:

            # 为当前服务链分配服务器，直到其降低到时延约束以下
            cur_delay = max_delay
            services = self.env.get_service_chain(user_from, user_to)
            flag = True    # 能否将当前服务链的时延降低到时延约束以下
            while cur_delay > self.env.delay_limit:
                if len(services) == 0:
                    flag = False
                    break

                services.sort(key=lambda x: x.reduction_of_delay_when_add_a_server(), reverse=True)
                selected_service = services[0]

                if not self.allocate(selected_service, 1):
                    services.remove(selected_service)
                    continue
                else:
                    cur_delay = self.env.compute_interactive_delay(user_from, user_to)
                    self.record("allocation_for_limit", service=selected_service,
                                edge_node=self.env.edge_node_list[selected_service.node_id], num=1)

            if not flag:
                success = False
                break

            user_from, user_to, max_delay = self.env.compute_max_interactive_delay_by_given_user(cur_user,
                                                                                                 self.assigned_users)
        return success

    """
        如果容量足够，为Service增加若干服务器。
        返回true/false表示成功与否
    """
    def allocate(self, service: Service, num: int) -> bool:
        success = False
        edge_node = self.env.edge_node_list[service.node_id]
        if self.is_capacity_enough(service, edge_node, num):
            service.update_num_server(service.num_server + num, update_queuing_delay=True)
            edge_node.num_server += num
            assert edge_node.num_server <= edge_node.capacity, "EdgeNode {} out of capacity.".format(edge_node.node_id)
            success = True

        return success

    """
        记录操作: assignment / allocation
        ("assignment", service, edge_node, _)
        ("allocation_for_init", service, edge_node, num)
        ("allocation_for_limit", service, edge_node, num)
    """
    def record(self, op: str, service: Service, edge_node: EdgeNode, num: int = None):
        if op == "assignment":
            self.operation_records.append((op, service, edge_node, None))
        elif op == "allocation_for_init" or op == "allocation_for_limit":
            self.operation_records.append((op, service, edge_node, num))
        else:
            print("Unknown operation")

    """
        根据operation_records，反向撤销已执行的操作.

        特别说明：
        1. 对于 ”allocation_for_init“ 撤销操作，在更新service的服务器数量的时候，不要重新计算 queuing_delay，因为会触发除零异常。
           queuing_delay 的重置在撤销 ”assignment“时进行（对于一个用户的A/R服务，”allocation_for_init“ 和 ”assignment“是成对出现的）
        2. 对于 ”allocation_for_limit“，需要更新 queuing_delay
    """
    def undo_operations(self):
        while len(self.operation_records) > 0:
            op, service, edge_node, num = self.operation_records[-1]

            if op == "allocation_for_init":
                service.update_num_server(service.num_server - num, update_queuing_delay=False)
                edge_node.num_server -= num

            elif op == "allocation_for_limit":
                service.update_num_server(service.num_server - num, update_queuing_delay=True)
                edge_node.num_server -= num

            elif op == "assignment":
                service.reset()
                edge_node.service_list.pop((service.user_id, service.service_type))

            else:
                print("Unknown operations.")

            self.operation_records.pop(-1)

    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)