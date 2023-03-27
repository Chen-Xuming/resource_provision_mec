import copy

from alg_base import *

class GreedyAssignmentAllocation(BaseAlgorithm):
    def __init__(self, env, *args, **kwargs):
        BaseAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "min_cost_greedy" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        self.assigned_users = []    # 已经完成关联、服务器分配的用户

        # self.operation_records = []     # 按时间序记录assign/allocation的操作，用于undo

        self.debug_flag = True  # True = On, False = Off

    def run(self):
        self.start_time = time()

        # 为服务B选择EdgeNode: 选择初始化稳态条件开销最小的
        assert self.initialize_service_B(), "Initialize service B failed."

        # 对于每个用户，选择部署成本最小的方案
        self.deploy_for_users()

        self.get_running_time()
        self.compute_final_cost()
        self.DEBUG("cost = {}".format(self.final_cost))
        self.DEBUG("running_time = {}".format(self.running_time))


    """
        选择初始化稳态条件开销最小的EdgeNode作为service_b的站点
    """
    def initialize_service_B(self) -> bool:
        service_b = self.env.service_b

        min_initializing_cost = 1e10
        target_node = None
        num_server_needed = 0
        for edge_node in self.env.edge_node_list:   # type: EdgeNode
            if self.is_capacity_enough(service_b, edge_node, None):
                server_needed = service_b.get_num_server_for_stability(edge_node.service_rate[service_b.service_type])
                cost_needed = server_needed * edge_node.price[service_b.service_type]
                if cost_needed < min_initializing_cost:
                    min_initializing_cost = cost_needed
                    target_node = edge_node
                    num_server_needed = server_needed

        if target_node is not None:
            self.assign_and_initialize(service_b, target_node, records=None, num=num_server_needed)
            self.DEBUG("service B: node_id={}, service_rate={}".format(service_b.node_id, service_b.service_rate))
            return True
        else:
            return False

    """
        对于每个用户，选择部署成本最小的方案。
        对于每个用户，尝试每种可行的组合 （node_a, node_r），挑选出cost最小的方案。
    """
    def deploy_for_users(self):
        for user in self.env.user_list:     # type: User
            self.assigned_users.append(user)

            candidate_pairs = []
            for i in range(self.env.num_edge_node):
                for j in range(self.env.num_edge_node):
                    candidate_pairs.append((i, j))

            min_cost = 1e15
            best_operations = None    # 最佳的关联、分配方案
            while len(candidate_pairs) > 0:
                cur_operation_records = []       # 本轮的关联、分配方案

                pair = candidate_pairs[0]
                candidate_pairs.pop(0)

                node_a = self.env.edge_node_list[pair[0]]
                node_r = self.env.edge_node_list[pair[1]]

                # 首先要满足 Tx + Tp < T_limit，否则无论如何增加服务器，都无济于事
                if not self.is_tx_tp_satisfied(user, node_a, node_r):
                    continue

                # 1. 如果容量能够满足稳态约束，则进行初始化
                # 2. 服务器分配，若无法满足时延约束，撤销已进行操作。
                #    计算cost，如果比min_cost小，则记录下来，并撤销已有操作。
                if self.is_capacity_enough(user.service_A, node_a) and self.is_capacity_enough(user.service_R, node_r):
                    # step 1
                    # 注意：如果node_a和node_r是同一个，那么分配完之后service_A之后，还要检查capacity
                    self.assign_and_initialize(user.service_A, node_a, records=cur_operation_records, num=None)
                    if not self.is_capacity_enough(user.service_R, node_r):
                        self.undo_operations(cur_operation_records)
                        continue
                    else:
                        self.assign_and_initialize(user.service_R, node_r, records=cur_operation_records, num=None)

                    # step 2
                    if not self.allocate_for_delay_limitations(user, cur_operation_records):
                        self.undo_operations(cur_operation_records)
                    else:
                        cur_cost = self.env.compute_cost(assigned_user_list=self.assigned_users)
                        if cur_cost < min_cost:
                            min_cost = cur_cost
                            best_operations = copy.copy(cur_operation_records)

                        self.undo_operations(cur_operation_records)

            assert best_operations is not None, "No any capacity for user {}.".format(user.user_id)

            # 提交所选出的开销最小的方案
            self.commit(best_operations)

            self.DEBUG("[assign] user:{} to ({}, {}, {}), cur_cost = {}".format(user.user_id,
                                                                                user.service_A.node_id,
                                                                                user.service_B.node_id,
                                                                                user.service_R.node_id,
                                                                                min_cost))


    """
        将service关联到给定的EdgeNode，并分配若干服务器，
        records = None 表示不用记录，否则记录到给定的record列表
        num = None 表示初始化满足稳态条件
    """
    def assign_and_initialize(self, service: Service, edge_node: EdgeNode, records=None, num=None):
        num_server = num if num is not None \
                         else service.get_num_server_for_stability(edge_node.service_rate[service.service_type])

        service.node_id = edge_node.node_id
        service.service_rate = edge_node.service_rate[service.service_type]
        service.price = edge_node.price[service.service_type]
        service.update_num_server(num_server, update_queuing_delay=True)

        edge_node.service_list[(service.user_id, service.service_type)] = service
        edge_node.num_server += num_server
        assert edge_node.num_server <= edge_node.capacity, "EdgeNode {}: Out of capacity limitation.".format(edge_node.node_id)

        if records is not None:
            self.record(operation_records=records, op="assignment", service=service, edge_node=edge_node)
            self.record(operation_records=records, op="allocation_for_init", service=service, edge_node=edge_node, num=num_server)

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
        分配服务器以满足时延约束. 每次选择 reduction / price 最大的
    """
    def allocate_for_delay_limitations(self, cur_user: User, records: list) -> bool:
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

                services.sort(key=lambda x: x.reduction_of_delay_when_add_a_server() / x.price, reverse=True)
                selected_service = services[0]

                if not self.allocate(selected_service, 1):
                    services.remove(selected_service)
                    continue
                else:
                    cur_delay = self.env.compute_interactive_delay(user_from, user_to)
                    self.record(operation_records=records, op="allocation_for_limit", service=selected_service,
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
    def record(self, operation_records: list, op: str, service: Service, edge_node: EdgeNode, num: int = None):
        if op == "assignment":
            operation_records.append((op, service, edge_node, None))
        elif op == "allocation_for_init" or op == "allocation_for_limit":
            operation_records.append((op, service, edge_node, num))
        else:
            print("Unknown operation")

    """
        根据operation_records，反向撤销已执行的操作.

        特别说明：
        1. 对于 ”allocation_for_init“ 撤销操作，在更新service的服务器数量的时候，不要重新计算 queuing_delay，因为会触发除零异常。
           queuing_delay 的重置在撤销 ”assignment“时进行（对于一个用户的A/R服务，”allocation_for_init“ 和 ”assignment“是成对出现的）
        2. 对于1，一个例外是 Service B，因为 Service B 没有撤销 ”assignment“ 的操作。
        3. 对于 ”allocation_for_limit“，需要更新 queuing_delay
    """
    def undo_operations(self, operation_records: list):
        while len(operation_records) > 0:
            op, service, edge_node, num = operation_records[-1]

            if op == "allocation_for_init":
                should_update_delay = True if service.service_type == "B" else False
                service.update_num_server(service.num_server - num, update_queuing_delay=should_update_delay)
                edge_node.num_server -= num

            elif op == "allocation_for_limit":
                service.update_num_server(service.num_server - num, update_queuing_delay=True)
                edge_node.num_server -= num

            elif op == "assignment":
                service.reset()
                edge_node.service_list.pop((service.user_id, service.service_type))

            else:
                print("Unknown operations.")

            operation_records.pop(-1)

    """
        提交给定的根据给定的operation_records（按顺序执行所记录的操作）
    """
    def commit(self, operation_records: list):
        for record in operation_records:
            op, service, edge_node, num = record

            if op == "allocation_for_init" or op == "allocation_for_limit":
                service.update_num_server(service.num_server + num, update_queuing_delay=True)
                edge_node.num_server += num

            elif op == "assignment":
                service.node_id = edge_node.node_id
                service.service_rate = edge_node.service_rate[service.service_type]
                service.price = edge_node.price[service.service_type]
                edge_node.service_list[(service.user_id, service.service_type)] = service

            else:
                print("Unknown operations.")


    def DEBUG(self, info: str):
        if self.debug_flag:
            print(info)
