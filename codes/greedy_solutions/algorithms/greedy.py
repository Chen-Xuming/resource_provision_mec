import copy
from codes.greedy_solutions.algorithms.base import *

class GreedyAssignmentAllocation(BaseAlgorithm):
    def __init__(self, env, *args, **kwargs):
        BaseAlgorithm.__init__(self, env, *args, **kwargs)
        self.algorithm_name = "min_cost_greedy" if "algorithm_name" not in kwargs else kwargs["algorithm_name"]

        self.assigned_users = []    # 已经完成关联、服务器分配的用户

        self.debug_flag = False  # True = On, False = Off

        # 初始化服务B位置的方法
        # min_sum_tx = 到其它节点的时延和最小
        # min_init_cost = 初始化满足稳态条件所需开销最小
        self.init_B_alg = "min_sum_tx"

        self.solution = []  # 算法的解

    def run(self):
        self.start_time = time()

        # 为服务B选择EdgeNode
        self.initialize_service_B()

        # 对于每个用户，选择部署成本最小的方案
        success = self.deploy_for_users()
        if not success:
            print("Invalid Solution.")
            return False

        self.get_running_time()
        self.compute_final_cost()

        self.DEBUG("cost = {}".format(self.final_cost))
        self.DEBUG("running_time = {}".format(self.running_time))

        return True

    def get_solution(self):
        return self.solution, self.final_cost, self.avg_delay, self.running_time

    def initialize_service_B(self):
        if self.init_B_alg == "min_sum_tx":
            self.initialize_service_B_min_sum_tx()

        elif self.init_B_alg == "min_init_cost":
            self.initialize_service_B_min_cost()

        else:
            raise Exception("Unknown initialization algorithm for service B.")


    """
        选择初始化稳态条件开销最小的EdgeNode作为service_b的站点
    """
    def initialize_service_B_min_cost(self):
        service_b = self.env.service_b

        min_initializing_cost = 1e10
        target_node = None
        for edge_node in self.env.edge_node_list:   # type: EdgeNode
            cost_needed = 0

            server_needed = service_b.get_num_server_for_stability(edge_node.service_rate[service_b.service_type])
            extra_server_needed = server_needed - edge_node.capacity
            if extra_server_needed > 0:
                cost_needed += edge_node.capacity * edge_node.price[service_b.service_type]
                cost_needed += extra_server_needed * edge_node.extra_price[service_b.service_type]
            else:
                cost_needed = server_needed * edge_node.price[service_b.service_type]

            if cost_needed < min_initializing_cost:
                min_initializing_cost = cost_needed
                target_node = edge_node

        self.assign_and_initialize(service_b, target_node, records=None)
        self.DEBUG("service B: node_id={}, service_rate={}".format(service_b.node_id, service_b.service_rate))


    """
        选择到其它节点时延之和最小的节点，当作服务B的关联节点
    """
    def initialize_service_B_min_sum_tx(self):
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

        self.assign_and_initialize(service_b, target_node, records=None)
        self.DEBUG("service B: node_id={}, service_rate={}".format(service_b.node_id, service_b.service_rate))


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

                self.assign_and_initialize(user.service_A, node_a, records=cur_operation_records)
                self.assign_and_initialize(user.service_R, node_r, records=cur_operation_records)

                # 分配服务器使交互时延降低至 T_limit 以下
                self.allocate_for_delay_limitations(cur_operation_records, user)

                # 1. 计算cost，如果小于min_cost. 则记录 best_operations 为当前的操作
                # 2. undo当前操作
                cur_cost = self.env.compute_cost()
                if cur_cost < min_cost:
                    min_cost = cur_cost
                    best_operations = copy.copy(cur_operation_records)

                self.undo_operations(operation_records=cur_operation_records)


            # 提交所选出的开销最小的方案（若存在）
            if best_operations is None:
                return False
            self.commit(best_operations)

            self.DEBUG("[assign] user:{} to ({}, {}, {}), cur_cost = {}".format(user.user_id,
                                                                                user.service_A.node_id,
                                                                                user.service_B.node_id,
                                                                                user.service_R.node_id,
                                                                                min_cost))

            # 记录当前用户的解
            self.solution.append((user.service_A.node_id, user.service_B.node_id, user.service_R.node_id))

        return True

    """
        将service关联到给定的EdgeNode，并分配若干服务器，初始化满足稳态条件
    """
    def assign_and_initialize(self, service: Service, edge_node: EdgeNode, records: list = None, user=None):
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

        if records is not None:
            self.record(operation_records=records, op="assignment", service=service, edge_node=edge_node, user=user)
            self.record(operation_records=records, op="allocation_for_init", service=service, edge_node=edge_node, num=num_server,
                        num_extra=extra_num_server)

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
        分配服务器以满足时延约束. 每次选择 reduction / price 最大的
    """
    # FIXME: x.price 没有考虑 x.extra_price
    def allocate_for_delay_limitations(self, records: list, cur_user: User):
        user_from, user_to, max_delay = self.env.compute_max_interactive_delay_by_given_user(cur_user, self.assigned_users)
        while max_delay > self.env.delay_limit:
            # 为当前服务链分配服务器，直到其降低到时延约束以下
            services = self.env.get_service_chain(user_from, user_to)
            services.sort(key=lambda x: x.reduction_of_delay_when_add_a_server() / x.price, reverse=True)
            selected_service = services[0]
            self.allocate(selected_service, 1, records)

            user_from, user_to, max_delay = self.env.compute_max_interactive_delay_by_given_user(cur_user, self.assigned_users)

    """
        如果容量足够，为Service增加若干服务器。
        返回true/false表示成功与否
    """
    def allocate(self, service: Service, num: int, records: list):
        edge_node = self.env.edge_node_list[service.node_id]  # type:EdgeNode
        extra_num = self.compute_num_extra_server(num, edge_node)

        edge_node.num_server += num
        edge_node.num_extra_server += extra_num
        service.update_num_server(service.num_server + num, service.num_extra_server + extra_num,
                                  update_queuing_delay=True)

        self.record(operation_records=records, op="allocation_for_limit", service=service,
                    edge_node=edge_node, num=1, num_extra=extra_num)

    """
        记录操作: assignment / allocation
        ("assignment", service, edge_node)
        ("allocation_for_init", service, edge_node, num, num_extra)
        ("allocation_for_limit", service, edge_node, num, num_extra)
    """
    def record(self, operation_records: list, op: str, service: Service, edge_node: EdgeNode = None, num: int = None,
               num_extra: int = None, user: User = None):
        if op == "assignment":
            operation_records.append((op, service, edge_node, user))
        elif op == "allocation_for_init" or op == "allocation_for_limit":
            operation_records.append((op, service, edge_node, num, num_extra))
        else:
            print("[record]: Unknown operations: {}.".format(op))

    """
        根据operation_records，反向撤销已执行的操作.

        特别说明：
        1. 对于 ”allocation_for_init“ 撤销操作，在更新service的服务器数量的时候，不要重新计算 queuing_delay，因为会触发除零异常。
           queuing_delay 的重置在撤销 ”assignment“时进行（对于一个用户的A/R服务，”allocation_for_init“ 和 ”assignment“是成对出现的）
        2. 对于 ”allocation_for_limit“，需要更新 queuing_delay
    """
    def undo_operations(self, operation_records: list):
        while len(operation_records) > 0:
            op = operation_records[-1][0]

            if op == "allocation_for_init":
                op, service, edge_node, num, num_extra = operation_records[-1]
                service.update_num_server(service.num_server - num, service.num_extra_server - num_extra, update_queuing_delay=False)
                edge_node.num_server -= num
                edge_node.num_extra_server -= num_extra

            elif op == "allocation_for_limit":
                op, service, edge_node, num, num_extra = operation_records[-1]
                service.update_num_server(service.num_server - num, service.num_extra_server - num_extra, update_queuing_delay=True)
                edge_node.num_server -= num
                edge_node.num_extra_server -= num_extra

            elif op == "assignment":
                op, service, edge_node = operation_records[-1][0], operation_records[-1][1], operation_records[-1][2]
                service.reset()
                edge_node.service_list.pop((service.user_id, service.service_type))

            else:
                print("[undo_operations]: Unknown operations: {}.".format(op))

            operation_records.pop(-1)

    """
        提交给定的根据给定的operation_records（按顺序执行所记录的操作）
    """
    def commit(self, operation_records: list):
        for record in operation_records:
            op = record[0]

            if op == "allocation_for_init" or op == "allocation_for_limit":
                op, service, edge_node, num, num_extra = record
                service.update_num_server(service.num_server + num, service.num_extra_server + num_extra, update_queuing_delay=True)
                edge_node.num_server += num
                edge_node.num_extra_server += num_extra

            elif op == "assignment":
                op, service, edge_node = record[0], record[1], record[2]
                service.node_id = edge_node.node_id
                service.service_rate = edge_node.service_rate[service.service_type]
                service.price = edge_node.price[service.service_type]
                service.extra_price = edge_node.extra_price[service.service_type]
                edge_node.service_list[(service.user_id, service.service_type)] = service

            else:
                print("[commit]: Unknown operations: {}.".format(op))


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
