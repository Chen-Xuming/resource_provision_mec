import numpy as np
import math

class Instance:
    def __init__(self, id):
        self._id = id
        self._groups = []

    def __str__(self):
        return 'instance: {} groups: {}'.format(self._id, self._groups)

class ShareViewGroup:
    def __init__(self, id, instance_id):
        self._id = id
        self._instance_id = instance_id
        self._users = []

    def __str__(self):
        return 'instance: {} group: {} users: {}'.format(self._instance_id, self._id, self._users)

class User:
    def __init__(self, id, instance_id, group_id):
        self._id = id
        self._instance_id = instance_id
        self._group_id = group_id
        self._arrival_rate = 0

        # 逻辑服务，渲染服务
        self._service_a = 0
        self._service_r = 0
        self._sub_service_r = 0  # R服务中的子服务

        def __str__(self):
            return 'instance: {} group: {} user: {} arrival_rate:{} service a:{} r:{} sub_r:{}'.format(
                self._instance_id,
                self._group_id,
                self._id, self._arrival_rate,
                self._service_a,
                self._service_r,
                self._sub_service_r)

"""
    1. 编号方式：每类服务的id从0开始
    2. 子服务编号：每个服务的子服务的id从0开始
"""
class Service:
    def __init__(self, id, type):
        self._id = id
        self._type = type
        self._service_rate = 0
        self._arrival_rate = 0
        self._users = []

        # self._group = set()     # 可能没用了

        self._price = 0
        self._num_server = 1
        self._queuing_delay = 0.

        self._initial_max_crossing_delay = 0.   # env初始化后，经过这个服务的最大交互时延

        # 仅R类服务时有效
        self._sub_id = None

    def __str__(self):
        return 'service: {} {},{} service_rate: {} arrival_rate: {} users:{} price:{} num_server:{} arr/ser = {} queuing_delay:{} delay_reduction: {}'.format(
            self._type,
            self._id,
            self._sub_id,
            self._service_rate,
            self._arrival_rate,
            self._users,
            self._price,
            self._num_server,
            self._arrival_rate / self._service_rate,
            self.compute_queuing_delay(self._num_server),
            self.reduction_of_delay_when_add_a_server())

    def get_info(self):
        return (self._type,
                self._id,
                self._sub_id,
                round(self._service_rate, 3),
                round(self._arrival_rate, 3),
                self._price,
                self._num_server,
                round(self._arrival_rate / self._service_rate, 3),
                round(self.compute_queuing_delay(self._num_server) * 1000, 2),
                round(self.reduction_of_delay_when_add_a_server() * 1000, 2))

    # 更新服务器的数量，以及排队时延
    def update_num_server(self, n):
        self._num_server = n
        self._queuing_delay = self.compute_queuing_delay(self._num_server)

    # 初始化服务器个数为刚好达到稳态条件的个数
    def initialize_num_server(self):
        num_server = self._num_server  # 初始化是1
        while num_server * self._service_rate <= self._arrival_rate:
            num_server += 1
        self.update_num_server(num_server)

    # 增加一台服务器带来的时延减少量
    def reduction_of_delay_when_add_a_server(self):
        num_server = self._num_server + 1
        reduction = self._queuing_delay - self.compute_queuing_delay(num_server)
        return reduction

    # 增加若干台服务器带来的时延减少量
    def reduction_of_delay_when_add_some_server(self, n):
        if n == 0:
            return 0.
        num_server = self._num_server + n
        reduction = self._queuing_delay - self.compute_queuing_delay(num_server)
        return reduction

    # 减少一台服务器增加的时延量
    # 减少后的数量是否少于初始化时的数量，由客户端检查
    def delay_increment_when_reduce_a_server(self):
        num_server = self._num_server - 1
        increment = self.compute_queuing_delay(num_server) - self._queuing_delay
        return increment

    # 计算排队时延
    def compute_queuing_delay(self, num_server):
        queuing_delay_iteratively = self.compute_queuing_delay_iteratively(num_server)
        # print(queuing_delay_iteratively)
        assert queuing_delay_iteratively >= 0.
        return queuing_delay_iteratively

    # 用迭代方法计算排队时延
    def compute_queuing_delay_iteratively(self, num_server):
        lam = float(self._arrival_rate)
        mu = float(self._service_rate)
        c = num_server
        r = lam / mu
        rho = r / c
        assert rho < 1

        p0c_2 = 1.
        n = 1
        p0c_1 = r / n
        n += 1
        while n <= c:
            p0c_2 += p0c_1
            p0c_1 *= r / n
            n += 1

        p0 = 1 / (p0c_1 / (1 - rho) + p0c_2)
        wq = p0c_1 * p0 / c / (mu * (1 - rho) ** 2)
        assert wq >= 0.
        return wq

"""
    config = {
        "num_instance": num_instance,
        "num_service_a": num_service_a,
        "num_service_r": num_service_r,
        "budget_addition": budget_addition,
        "num_group_per_instance": num_group_per_instance,
        "num_user_per_group": num_user_per_group,
        "min_arrival_rate": min_arrival_rate,
        "max_arrival_rate": max_arrival_rate,
        "min_service_rate": min_service_rate,
        "max_service_rate": max_service_rate,
        "min_price": min_price,
        "max_price": max_price,
        "trigger_probability": trigger_probability,
        "tx_ua_min": ..      # tx是传输时延
        "tx_ua_max": ..
        "tx_aq_min": ..
        "tx_aq_max": ..
        "tx_qr_mix": ..
        "tx_qr_max": ..
        "tx_ru_mix": ..
        "tx_ru_max": ..
    }
"""
class Env:
    def __init__(self, config, rng, seed_sequence):
        self._config_list = config

        self._seed_sequence = seed_sequence
        self._rng = rng
        self._num_instance = config["num_instance"]
        self._num_service_a = config["num_service_a"]
        self._num_service_r = config["num_service_r"]

        # todo: 引入了子服务之后，service的数量与group有关
        self._num_service = 0

        self._num_group_per_instance = config["num_group_per_instance"]  # 每个instance的组数
        self._num_user_per_group = config["num_user_per_group"]  # 每个组的用户数

        self._max_arrival_rate = config["max_arrival_rate"]  # 61
        self._min_arrival_rate = config["min_arrival_rate"]

        self._max_price = config["max_price"]
        self._min_price = config["min_price"]

        self._trigger_probability = config["trigger_probability"]

        self._cost_budget = 0       # 初始化开销 + 预算
        self._budget_addition = config["budget_addition"]

        # 传输时延取值范围
        self._tx_u_a_minmax = (config["tx_ua_min"], config["tx_ua_max"])
        self._tx_a_q_minmax = (config["tx_aq_min"], config["tx_aq_max"])
        self._tx_q_r_minmax = (config["tx_qr_min"], config["tx_qr_max"])
        self._tx_r_u_minmax = (config["tx_ru_min"], config["tx_ru_max"])

        self.initialize_user()

        # 去除服务C后，这一步骤不再需要了
        # self.correct_initialize_user()

        self.initialize_service()

        # self.initialize_num_server()

        # 当只考虑排队时延时，初始化这个
        # self.get_queuing_delay_without_duplicate()

        # 当考虑整个交互时延时，初始化这个
        self.get_interaction_delay_without_duplicate()

        self.initialize_num_server()

    """
        初始化用户
    """
    def initialize_user(self):
        self._instances = []
        self._groups = []
        self._users = []

        instance_id = 0
        group_id = 0
        user_id = 0

        self._num_user = self._num_instance * self._num_group_per_instance * self._num_user_per_group
        self.user_service_a = np.zeros((self._num_user, self._num_service_a))
        self.user_service_r = np.zeros((self._num_user, self._num_service_r))

        for i in range(self._num_instance):
            instance = Instance(instance_id)
            for g in range(self._num_group_per_instance):
                group = ShareViewGroup(group_id, instance_id)
                for u in range(self._num_user_per_group):
                    user = User(user_id, instance_id, group_id)
                    user._arrival_rate = self._rng.integers(self._min_arrival_rate, self._max_arrival_rate + 1)     # [min_arrival_rate, max_arrival_rate]

                    user._service_a = self._rng.integers(0, self._num_service_a)
                    user._service_r = self._rng.integers(0, self._num_service_r)

                    # todo: ------------------ 子服务 --------------------
                    # 用户的子服务分配在 initialize_service() 部分

                    # 二维矩阵保存关联关系
                    self.user_service_a[user._id][user._service_a] = 1
                    self.user_service_r[user._id][user._service_r] = 1

                    self._users.append(user)
                    group._users.append(user_id)
                    user_id += 1
                self._groups.append(group)
                instance._groups.append(group_id)
                group_id += 1
            self._instances.append(instance)
            instance_id += 1

    def initialize_service(self):
        # 传输时延
        # self._tx_u_a = self._rng.integers(self._min_tx, self._max_tx + 1, (self._num_user, self._num_service_a))
        # self._tx_a_b0 = self._rng.integers(self._min_tx, self._max_tx + 1, self._num_service_a)
        # self._tx_b0_b = self._rng.integers(self._min_tx, self._max_tx + 1, self._num_service_b)
        # self._tx_b_c = self._rng.integers(self._min_tx, self._max_tx + 1, (self._num_service_b, self._num_service_c))
        # self._tx_c_r = self._rng.integers(self._min_tx, self._max_tx + 1, (self._num_service_c, self._num_service_r))
        # self._tx_r_u = self._rng.integers(self._min_tx, self._max_tx + 1, (self._num_service_r, self._num_user))

        # 传输时延(用户--逻辑服务器a--同步服务器q--渲染服务器r--用户)
        self._tx_u_a = self._rng.integers(self._tx_u_a_minmax[0], self._tx_u_a_minmax[1] + 1, (self._num_user, self._num_service_a))
        self._tx_a_q = self._rng.integers(self._tx_a_q_minmax[0], self._tx_a_q_minmax[1] + 1, self._num_service_a)
        self._tx_q_r = self._rng.integers(self._tx_q_r_minmax[0], self._tx_q_r_minmax[1] + 1, self._num_service_r)
        self._tx_r_u = self._rng.integers(self._tx_r_u_minmax[0], self._tx_r_u_minmax[1] + 1, (self._num_service_r, self._num_user))

        self._service_A = []
        self._service_R = []        # [[sub-services], [sub-services],...]

        #  ------------------ service_A -----------------------
        for a in range(self._num_service_a):
            service = Service(a, 'a')
            # service._service_rate = self._rng.integers(self._min_service_rate, self._max_service_rate + 1)
            service._service_rate = self._rng.integers(100, 200 + 1)
            for u in range(self._num_user):
                if self.user_service_a[u][a] == 1:
                    service._arrival_rate += self._users[u]._arrival_rate
                    service._users.append(u)

            service._price = self._rng.integers(self._min_price, self._max_price + 1)
            # service._price = service._service_rate / self._min_service_rate * 2 + 1

            self._service_A.append(service)


        # --------------------- service_q --------------------
        # 同步服务器q，仅此一个
        self._service_q = Service(0, 'q')
        # self._service_b0._service_rate = self._rng.integers(self._min_service_rate, self._max_service_rate + 1)
        self._service_q._service_rate = self._rng.integers(200, 500 + 1)
        for u in range(self._num_user):
            self._service_q._arrival_rate += self._users[u]._arrival_rate
            self._service_q._users.append(u)

        self._service_q._price = self._rng.integers(self._min_price, self._max_price + 1)
        # self._service_b0._price = self._service_b0._service_rate / self._min_service_rate * 2 + 1


        # --------------------- service_R ----------------------
        for r in range(self._num_service_r):
            sub_service_list = []

            # ----找出站点r上有哪些用户组-----
            share_group = {}
            for u in range(self._num_user):
                if self.user_service_r[u][r] == 1:
                    if self._users[u]._group_id not in share_group:
                        share_group[self._users[u]._group_id] = [u]
                    else:
                        share_group[self._users[u]._group_id].append(u)

            # 同一个站点的每个子服务的服务器服务率、价格是一样的
            price = self._rng.integers(self._min_price, self._max_price + 1)
            # service_rate = self._rng.integers(self._min_service_rate, self._max_service_rate + 1)
            service_rate = self._rng.integers(40, 50 + 1)
            # price = service_rate / self._min_service_rate * 2 + 1

            # ----- 为每一个用户组新建一个子服务 ----------
            sub_service_id = 0
            for group, users in share_group.items():
                sub_service = Service(r, 'r')
                sub_service._sub_id = sub_service_id
                sub_service._users = users
                sub_service._service_rate = service_rate
                sub_service._price = price

                # 子服务的到达率是 λ * p
                sub_service._arrival_rate = self._service_q._arrival_rate * self._trigger_probability

                for user_id in users:
                    self._users[user_id]._sub_service_r = sub_service_id

                sub_service_list.append(sub_service)
                sub_service_id += 1
            self._service_R.append(sub_service_list)

        # ----------------------------- 服务A，q，R初始化完成 ----------------------------------

        # 计算服务个数
        num_service_r = 0
        for sub_service_list in self._service_R:
            num_service_r += len(sub_service_list)
        self._num_service = self._num_service_a + 1 + num_service_r


    """
        获取去冗余后的排队时延对
        1. 由于C, R服务增加了子服务，每个子服务只为一个用户组服务，因此冗余的情况只会出现在同组用户中
    """
    def get_queuing_delay_without_duplicate(self):
        self._queuing_delay_without_duplicate = {}
        num = 0
        for i in range(self._num_user):
            for j in range(self._num_user):
                delay_index = self.get_delay_index(i, j)
                a, q, r, sub_r = self.get_service_index(i, j)

                service_set = [a, q, (r, sub_r)]    # 这里各个编号可能会相同，所以不能用集合

                exist = False
                for key, value in self._queuing_delay_without_duplicate.items():
                    if value['service_set'] == service_set:
                        exist = True
                        break
                if not exist:
                    self._queuing_delay_without_duplicate[delay_index] = {
                        "service_set": service_set,
                        "users": (i, j),
                        "new_index": num
                    }
                    num += 1

    """
        获取去冗余后的交互时延对：在get_queuing_delay_without_duplicate()的基础上进行更改
        1. 交互时延 = 传输 + 排队 + 处理
        2. 对于同一个服务链上的用户对，选择交互时延最大的那一对        
    """
    def get_interaction_delay_without_duplicate(self):
        self._interaction_delay_without_duplicate = {}
        num = 0
        for i in range(self._num_user):
            for j in range(self._num_user):
                delay_index = self.get_delay_index(i, j)
                a, q, r, sub_r = self.get_service_index(i, j)
                service_set = [a, q, (r, sub_r)]

                exist = False
                for key, value in self._interaction_delay_without_duplicate.items():
                    if value["service_chain"] == service_set:
                        exist = True

                        # 如果当前用户的交互时延比服务链上现有用户的交互时延大，就取代它
                        interaction_delay_i_j = self.compute_interaction_delay(i, j)
                        interaction_delay_old = self.compute_interaction_delay(value["users"][0], value["users"][1])
                        if interaction_delay_i_j > interaction_delay_old:
                            value["users"] = (i, j)
                        break

                if not exist:
                    self._interaction_delay_without_duplicate[delay_index] = {
                        "service_chain": service_set,
                        "users": (i, j),
                        "new_index": num
                    }
                    num += 1

    def get_delay_index(self, user_i, user_j):
        index = 0
        for i in range(self._num_user):
            for j in range(self._num_user):
                if user_i == i and user_j == j:
                    return index
                else:
                    index += 1
        return -1   # 多余

    # 获取时延对涉及的服务的编号(每一类服务编号从0开始)
    # a服务是user_i的， 其他服务是user_j的
    def get_service_index(self, user_i, user_j):
        a = self._users[user_i]._service_a
        q = 0  # q 是 同步服务

        r = self._users[user_j]._service_r
        sub_r = self._users[user_j]._sub_service_r

        return a, q, r, sub_r

    # 初始化服务器个数
    def initialize_num_server(self):
        for i in range(self._num_service_a):
            self._service_A[i].initialize_num_server()

        self._service_q.initialize_num_server()

        for i in range(self._num_service_r):
            # 对一个R站点，为其每一个子服务初始化服务器
            for sub_service in self._service_R[i]:
                sub_service.initialize_num_server()

        """
            记录以下信息：
            1. 最大开销（初始化开销 + 预算）
            2. 保存初始时各个服务的服务器个数（有些算法需要重置操作）
            3. 初始化时的最大时延
        """
        cost = self.compute_cost()
        self._cost_budget = cost + self._budget_addition
        self.save_initial_num_server()
        # max_delay, user_i, user_j = self.get_max_queuing_delay_in_interaction_delay()
        # self.initialize_max_delay = max_delay * 1000

    """
        重置各个服务的服务器个数为初始个数
    """
    def re_initialize_num_server(self):
        for i, service in enumerate(self._service_A):
            service.update_num_server(self._initial_num_server["service_a"][i])

        self._service_q.update_num_server(self._initial_num_server["service_q"])

        for i in range(self._num_service_r):
            for j, service in enumerate(self._service_R[i]):
                service.update_num_server(self._initial_num_server["service_r"][i][j])

    """
        计算开销
    """
    def compute_cost(self):
        cost = 0

        for service in self._service_A:
            cost += service._num_server * service._price

        cost += self._service_q._num_server * self._service_q._price

        # R服务开销：所有子服务开销的总和
        for i in range(self._num_service_r):
            for sub_service in self._service_R[i]:
                cost += sub_service._num_server * sub_service._price

        return cost

    """
        记录初始化后的服务器个数，用字典+列表存储
    """
    def save_initial_num_server(self):
        self._initial_num_server = {
            "service_a": [],
            "service_q": 0,
            "service_r": []    # [[], [], ...]
        }

        for service in self._service_A:
            self._initial_num_server["service_a"].append(service._num_server)

        self._initial_num_server["service_q"] = self._service_q._num_server

        for i in range(self._num_service_r):
            sub_service_num_servers = []
            for service in self._service_R[i]:
                sub_service_num_servers.append(service._num_server)
            self._initial_num_server["service_r"].append(sub_service_num_servers)

    """
        获取最大时延（排队时延），已去冗余
    """
    def get_max_queuing_delay_in_interaction_delay(self):
        self._queuing_delay_dict = {}
        for key, value in self._queuing_delay_without_duplicate.items():
            user_i = value['users'][0]
            user_j = value['users'][1]
            self._queuing_delay_dict[(user_i, user_j)] = self.compute_queuing_delay(user_i, user_j)

        max_user_pair = max(self._queuing_delay_dict, key=lambda k: self._queuing_delay_dict[k])
        return self._queuing_delay_dict[max_user_pair], max_user_pair[0], max_user_pair[1]

    """
        计算某对用户的排队时延
    """
    def compute_queuing_delay(self, user_i, user_j):
        queuing_delay = self._service_A[self._users[user_i]._service_a]._queuing_delay + \
                        self._service_q._queuing_delay + \
                        self._service_R[self._users[user_j]._service_r][self._users[user_j]._sub_service_r]._queuing_delay
        return queuing_delay

    """
        获取两对排队时延最大的用户对
    """
    def get_max_two_queuing_delay_in_interaction_delay(self):
        self._queuing_delay_dict = {}
        for key, value in self._queuing_delay_without_duplicate.items():
            user_i = value['users'][0]
            user_j = value['users'][1]
            self._queuing_delay_dict[(user_i, user_j)] = self.compute_queuing_delay(user_i, user_j)

        delay_tuple = zip(self._queuing_delay_dict.values(), self._queuing_delay_dict.keys())
        delay_list = sorted(delay_tuple)

        max_delay = delay_list[-1]  # i:max_delay[1][0], j:max_delay[1][1]
        _2nd_max_delay = delay_list[-2]
        return max_delay, _2nd_max_delay

    """
        计算用户对的交互时延（ = 传输时延 + 排队时延 + 处理时延）
    """
    def compute_interaction_delay(self, user_i, user_j):
        # 传输时延(ms)
        transmission_delay = self._tx_u_a[user_i][self._users[user_i]._service_a] + \
                             self._tx_a_q[self._users[user_i]._service_a] + \
                             self._tx_q_r[self._users[user_j]._service_r] + \
                             self._tx_r_u[self._users[user_j]._service_r][user_j]

        # 排队时延(s)
        queuing_delay = self.compute_queuing_delay(user_i, user_j)

        # 处理时延(s)
        processing_time = 1 / self._service_A[self._users[user_i]._service_a]._service_rate +\
                            1 / self._service_q._service_rate +\
                            1 / self._service_R[self._users[user_j]._service_r][self._users[user_j]._sub_service_r]._service_rate

        # 总时延(s)
        delay = transmission_delay / 1000 + queuing_delay + processing_time
        #
        # todo
        # delay = transmission_delay / 1000 / 10 + queuing_delay + processing_time / 10

        # print("-------------------------------------")
        # print("trans: ", transmission_delay / 1000)
        # print("process: ", processing_time)
        # print("trans + process: ", transmission_delay / 1000 + processing_time)
        # print("queue: ", queuing_delay)

        return delay

    """
        获取最大的交互时延
    """
    def get_max_interaction_delay(self):
        self._interaction_delay_dict = {}
        for key, value in self._interaction_delay_without_duplicate.items():
            user_i = value['users'][0]
            user_j = value['users'][1]
            self._interaction_delay_dict[(user_i, user_j)] = self.compute_interaction_delay(user_i, user_j)

        max_user_pair = max(self._interaction_delay_dict, key=lambda k: self._interaction_delay_dict[k])
        return self._interaction_delay_dict[max_user_pair], max_user_pair[0], max_user_pair[1]

    """
        获取最大的两个交互时延
    """
    def get_top_two_interaction_delay(self):
        self._interaction_delay_dict = {}
        for key, value in self._interaction_delay_without_duplicate.items():
            user_i = value["users"][0]
            user_j = value["users"][1]
            self._interaction_delay_dict[(user_i, user_j)] = self.compute_interaction_delay(user_i, user_j)

        delay_tuple = zip(self._interaction_delay_dict.values(), self._interaction_delay_dict.keys())
        delay_list = sorted(delay_tuple)

        max_delay = delay_list[-1]
        second_max_delay = delay_list[-2]

        return max_delay, second_max_delay

    """ ------------------------------------------------------------------
        debug
    """
    def check_users(self):
        instance = self._instances[0]
        for group_id in instance._groups:
            print("---------- group: {} ------------".format(group_id))
            for user_id in self._groups[group_id]._users:
                user = self._users[user_id]
                print("user: {}, arrival_rate: {}, services: {}".format(user_id, user._arrival_rate, (user._service_a, 0,
                                                                        (user._service_r, user._sub_service_r))))

    def check_services(self):
        print("\n---------- service A -------------")
        for service in self._service_A:
            print("service: ({}, {}), service_rate: {}, arrival_rate: {}, users: {}, price: {}, num_server: {}, queuinng_delay: {}"
                  .format(service._type,
                          service._id,
                          service._service_rate,
                          service._arrival_rate,
                          service._users,
                          service._price,
                          service._num_server,
                          service._queuing_delay))

        print("\n---------- service q -------------")
        service_q = self._service_q
        print("service: ({}, {}), service_rate: {}, arrival_rate: {}, users: {}, price: {}, num_server: {}, queuinng_delay: {}"
              .format(service_q._type,
                    service_q._id,
                    service_q._service_rate,
                    service_q._arrival_rate,
                    service_q._users,
                    service_q._price,
                    service_q._num_server,
                    service_q._queuing_delay))


        print("\n---------- service R -------------")
        for i in range(self._num_service_r):
            for service in self._service_R[i]:
                print("service: ({}, {}), service_rate: {}, arrival_rate: {}, users: {}, price: {}, num_server: {}, queuinng_delay: {}"
                      .format(service._type,
                              (service._id, service._sub_id),
                              service._service_rate,
                              service._arrival_rate,
                              service._users,
                              service._price,
                              service._num_server,
                              service._queuing_delay))

    """
        -- 保存初始化环境配置 --
        这里只保存了environment级别的信息，对于service和user的信息（及其关联关系）则没有保存，
        因为可以依靠所保存的随机种子进行还原。注意：如果各个依赖随机种子的属性生成顺序发生变化，那么所保存的随机种子将失效！
    """
    def save_config(self, save_path):
        config = dict()

        config["seed_entropy"] = self._seed_sequence.entropy
        # config["rng"] = self._rng
        config["num_instance"] = self._num_instance
        config["num_service_a"] = self._num_service_a
        config["num_service_r"] = self._num_service_r
        config["num_group_per_instance"] = self._num_group_per_instance
        config["num_user_per_group"] = self._num_user_per_group
        config["max_price"] = self._max_price
        config["min_price"] = self._min_price
        config["trigger_probability"] = self._trigger_probability
        config["budget_addition"] = self._budget_addition
        config["tx_ua_min"] = self._tx_u_a_minmax[0]
        config["tx_ua_max"] = self._tx_u_a_minmax[1]
        config["tx_aq_min"] = self._tx_a_q_minmax[0]
        config["tx_aq_max"] = self._tx_a_q_minmax[1]
        config["tx_qr_min"] = self._tx_q_r_minmax[0]
        config["tx_qr_max"] = self._tx_q_r_minmax[1]
        config["tx_ru_min"] = self._tx_r_u_minmax[0]
        config["tx_ru_max"] = self._tx_r_u_minmax[1]

        import simplejson as json
        with open(save_path, "a") as file:
            json.dump(config, file)


    """
        1. 获取服务列表
        2. 获取各个服务的信息，按照type和id排列
    """
    def get_services_and_infos(self):
        service_info_list = []
        service_list = []
        for service in self._service_A:
            service_list.append(service)
        service_list.append(self._service_q)
        for service in self._service_R:
            for sub_service in service:
                service_list.append(sub_service)
        self.computing_crossing_max_delay()

        for service in service_list:
            key = (service._type, service._id, service._sub_id)
            attr = [service._initial_max_crossing_delay,
                    service.reduction_of_delay_when_add_a_server(),
                    service._price,
                    service._service_rate,
                    service._arrival_rate]
            service_info_list.append([key, attr])

        return service_list, service_info_list

    def computing_crossing_max_delay(self):
        for key, value in self._interaction_delay_without_duplicate.items():
            user_i = value['users'][0]
            user_j = value['users'][1]
            delay_ij = self.compute_interaction_delay(user_i, user_j)
            a, q, r, sub_r = self.get_service_index(user_i, user_j)
            self._service_A[a]._initial_max_crossing_delay = max(self._service_A[a]._initial_max_crossing_delay, delay_ij)
            self._service_q._initial_max_crossing_delay = max(self._service_q._initial_max_crossing_delay, delay_ij)
            self._service_R[r][sub_r]._initial_max_crossing_delay = max(self._service_R[r][sub_r]._initial_max_crossing_delay, delay_ij)











