from codes.min_cost_v2.env.environment import *
import numpy as np

class PackagedEnv:
    def __init__(self, env: Environment):
        self.env = env

        self.num_edge_node = env.num_edge_node

        self.node_state_dim = 12

        self.assigned_users = []  # 已经完成关联、服务器分配的用户
        self.current_user_id = 0

        self.cost = 0

        self.debug_flag = False  # True = On, False = Off

        self.initialize()
        # print("Initialization finished.")

    """
        action[0] = service_a, action[1] = service_r.
        返回：(s_, mask_, r, done, _)
    """
    def step(self, action):
        if action is None:
            done = True
            mask = [False] * (self.num_edge_node * 2)
            full_mask = [False] * (self.num_edge_node * self.num_edge_node)
            reward = -5000      # FIXME:
            state = self.get_state(done=True, mask=mask)
            return state, mask, full_mask, reward, done, None


        reward = 0
        done = False

        user = self.env.user_list[self.current_user_id]
        self.assigned_users.append(user)

        node_a_id = action[0]
        node_r_id = action[1]
        node_a = self.env.edge_node_list[node_a_id]
        node_r = self.env.edge_node_list[node_r_id]

        # assert self.is_tx_tp_satisfied_2(user, node_a, node_r), "Illegal Action!"


        # 将user的服务A/R放置在指定位置，并初始化服务器个数
        self.assign_and_initialize(user.service_A, node_a)
        self.assign_and_initialize(user.service_R, node_r)

        # 分配服务器使交互时延降低至 T_limit 以下
        self.allocate_for_delay_limitations(user)

        # 计算cost的增量，以及reward
        new_cost = self.env.compute_cost(self.assigned_users)
        delta_cost = new_cost - self.cost
        self.cost = new_cost
        reward += -delta_cost

        # print("[Assign] user #{} ---> ({}, {}, {})".format(user.user_id, user.service_A.node_id, user.service_B.node_id, user.service_R.node_id))
        self.current_user_id += 1
        if self.current_user_id == self.env.num_user:
            done = True

        mask_ = None
        full_mask_ = None
        if self.current_user_id < self.env.num_user:
            mask_, full_mask_ = self.get_mask()
        else:
            mask_ = [False] * (self.num_edge_node * 2)
            full_mask_ = [False] * (self.num_edge_node * self.num_edge_node)

        state_ = self.get_state(done, mask_)
        return state_, mask_, full_mask_, reward, done, None


    """
        获取当前的状态，共12个。
        [ABR服务的服务率，ABR服务单价，超出容量时ABR服务的单价，已分配服务器个数/容量，cost1, cost2]
        
        特别说明：
        cost1 = 当前用户为 𝑢_𝑖，若把服务A放置在此站点，满足𝑇(𝑢_𝑖,𝑢_𝑗 )≤𝑇_𝑙𝑖𝑚𝑖𝑡 , ∀𝑢_𝑗 ∈ 𝑈_𝑎𝑠𝑠𝑖𝑔𝑛𝑒𝑑 所需要的开销
        cost2 = 当前用户为 𝑢_𝑖，若把服务R放置在此站点，满足𝑇(𝑢_𝑗,𝑢_𝑖 )≤𝑇_𝑙𝑖𝑚𝑖𝑡 , ∀𝑢_𝑗 ∈ 𝑈_𝑎𝑠𝑠𝑖𝑔𝑛𝑒𝑑 所需要的开销
        当 done = True 的时候，表示用户已经分配完成，此时不用计算cost了
        
        归一化：
        （1）3个服务率归一化
        （2）6个单价归一化
        （3）已分配服务器个数/容量：N个站点的这个值进行归一化
        （4）2个cost归一化
    """
    def get_state(self, done=False, mask=None):
        states = []
        mask_a = mask[:self.num_edge_node]
        mask_r = mask[self.num_edge_node:]
        for node in self.env.edge_node_list:    # type: EdgeNode
            node_state = list()

            # 服务率
            node_state.append(node.service_rate["A"])
            node_state.append(node.service_rate["B"])
            node_state.append(node.service_rate["R"])

            # 价格
            node_state.append(node.price["A"])
            node_state.append(node.price["B"])
            node_state.append(node.price["R"])
            node_state.append(node.extra_price["A"])
            node_state.append(node.extra_price["B"])
            node_state.append(node.extra_price["R"])

            # 负载
            node_state.append(node.num_server / node.capacity)

            # cost
            # TODO: 计算cost
            if done:
                node_state.append(0)
                node_state.append(0)
            else:
                if not mask_a[node.node_id]:
                    node_state.append(-1)       # -1 标记
                else:
                    pass

                if not mask_r[node.node_id]:
                    node_state.append(-1)
                else:
                    pass



                if not mask_a[node.node_id] or not mask_r[node.node_id]:
                    node_state.append(0)
                    node_state.append(0)
                else:
                    # TODO
                    node_state.append(1)
                    node_state.append(1)

            states.append(node_state)

        states = np.array(states, dtype=float)

        # 状态归一化
        states[:, :3] = self.normalize(states[:, :3])
        states[:, 3:9] = self.normalize(states[:, 3:9])
        states[:, 9:10] = self.normalize(states[:, 9:10])
        states[:, 10:12] = self.normalize(states[:, 10:12])

        return states

    # 最大值最小值归一化，x = (x - min) / (max - min)，x 处于 [0, 1] 区间
    def normalize(self, data):
        max_num = np.max(data)
        min_num = np.min(data)
        if max_num == min_num:
            data = 0.
        else:
            data = (data - min_num) / (max_num - min_num)
        return data

    """
        为下一个用户计算各个站点的mask
    """
    def get_mask(self):
        cur_user = self.env.user_list[self.current_user_id]

        # 检查 Tx+Tp(cur_user, other_users) 是否满足约束
        mask_a = []
        for node in self.env.edge_node_list:    # type: EdgeNode
            m = self.is_tx_tp_satisfied(cur_user, node, service_type="A")
            mask_a.append(m)

        # 检查 Tx+Tp(other_users, cur_user) 是否满足约束
        mask_r = []
        for node in self.env.edge_node_list:  # type: EdgeNode
            m = self.is_tx_tp_satisfied(cur_user, node, service_type="R")
            mask_r.append(m)

        """
            mask_a, mask_r 并没有考虑 cur_user 跟自己交互的情况，因为在分别考虑 a/r 的时候，自己的 r/a 并没有确定
            因此设置 full_mask(size = num_node * num_node)，包含任意组合的合法性
        """
        full_mask = []
        for node_a_id, ma in enumerate(mask_a):
            for node_r_id, mr in enumerate(mask_r):
                if not ma or not mr:
                    full_mask.append(False)
                else:
                    # ma, mr 都是 True 的时候，检查cur_user自己到自己的交互时延是否合法
                    full_mask.append(self.tx_tp_self_check(cur_user, self.env.edge_node_list[node_a_id],
                                                           self.env.edge_node_list[node_r_id]))

        return mask_a + mask_r, full_mask

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


    def is_tx_tp_satisfied(self, cur_user: User, node: EdgeNode, service_type) -> bool:
        flag = True

        if service_type == 'A':
            cur_user.service_A.node_id = node.node_id
            cur_user.service_A.service_rate = node.service_rate[cur_user.service_A.service_type]
            for u in self.assigned_users:   # type: User
                tx_tp = self.env.compute_tx_tp(cur_user, u)
                if tx_tp >= self.env.delay_limit:
                    flag = False
                    break
            cur_user.service_A.reset()

        elif service_type == 'R':
            cur_user.service_R.node_id = node.node_id
            cur_user.service_R.service_rate = node.service_rate[cur_user.service_R.service_type]
            for u in self.assigned_users:   # type: User
                tx_tp = self.env.compute_tx_tp(u, cur_user)
                if tx_tp >= self.env.delay_limit:
                    flag = False
                    break
            cur_user.service_R.reset()

        else:
            raise Exception("Unknown EdgeNode type.")

        return flag

    """
        计算与给定用户相关的时延对的 Tx + Tp，查看是否都小于 T_limit
    """
    def is_tx_tp_satisfied_2(self, user: User, node_a: EdgeNode, node_r: EdgeNode) -> bool:
        user.service_A.node_id = node_a.node_id
        user.service_A.service_rate = node_a.service_rate[user.service_A.service_type]
        user.service_R.node_id = node_r.node_id
        user.service_R.service_rate = node_r.service_rate[user.service_R.service_type]

        flag = True
        for u in self.assigned_users:  # type: User
            tx_tp = self.env.compute_tx_tp(user, u)
            if tx_tp >= self.env.delay_limit:
                flag = False
                print("User-pair({}, {}) not satisfied".format(user.user_id, u.user_id))
                break

            tx_tp = self.env.compute_tx_tp(u, user)
            if tx_tp >= self.env.delay_limit:
                flag = False
                print("User-pair({}, {}) not satisfied".format(u.user_id, user.user_id))
                break

        user.service_A.reset()
        user.service_R.reset()
        return flag

    def tx_tp_self_check(self, cur_user: User, node_a: EdgeNode, node_r: EdgeNode) -> bool:
        cur_user.service_A.node_id = node_a.node_id
        cur_user.service_R.node_id = node_r.node_id
        cur_user.service_A.service_rate = node_a.service_rate[cur_user.service_A.service_type]
        cur_user.service_R.service_rate = node_r.service_rate[cur_user.service_R.service_type]

        flag = True
        tx_tp = self.env.compute_tx_tp(cur_user, cur_user)
        if tx_tp >= self.env.delay_limit:
            flag = False

        cur_user.service_A.reset()
        cur_user.service_R.reset()
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