import numpy as np
from numpy.random import default_rng, SeedSequence
import math
from user_assignment import assign as assign_users

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
        "min_arrival_rate": min_arrival_rate,       # 用户到达率的范围
        "max_arrival_rate": max_arrival_rate,
        "min_service_rate_a": ...,                  # A服务的服务率范围
        "max_service_rate_a": ...,
        "min_service_rate_q": ...,                  # q服务的服务率范围
        "max_service_rate_q": ...,
        "min_service_rate_r": ...,                  # R服务的服务率范围
        "max_service_rate_r": ...,
        "trigger_probability": trigger_probability,
        "tx_ua_min": ..      # tx是传输时延
        "tx_ua_max": ..
        "tx_aq_min": ..
        "tx_aq_max": ..
        "tx_qr_mix": ..
        "tx_qr_max": ..
        "tx_ru_mix": ..
        "tx_ru_max": ..
        "assignment_algorithm":...     # 关联关系决策算法
    }
"""
class Env:
    def __init__(self, config: dict, rng: default_rng, seed_sequence: SeedSequence):
        self._config_list = config

        self._seed_sequence = seed_sequence
        self._rng = rng
        self._num_instance = config["num_instance"]
        self._num_service_a = config["num_service_a"]
        self._num_service_r = config["num_service_r"]

        # 保存 Instance, ShareViewGroup, User 实例
        # 在 initialize_user() 中初始化
        self._instances = None
        self._groups = None
        self._users = None

        # 引入了子服务之后，service的数量与group有关
        # 在 initialize_service() 中初始化
        self._num_service = 0

        self._num_group_per_instance = config["num_group_per_instance"]  # 每个instance的组数
        self._num_user_per_group = config["num_user_per_group"]  # 每个组的用户数

        # 系统中的用户数量
        self._num_user = self._num_instance * self._num_group_per_instance * self._num_user_per_group

        self._max_arrival_rate = config["max_arrival_rate"]
        self._min_arrival_rate = config["min_arrival_rate"]

        self._max_price = config["max_price"]
        self._min_price = config["min_price"]

        self._trigger_probability = config["trigger_probability"]

        self._cost_budget = 0       # 初始化开销 + 预算
        self._budget_addition = config["budget_addition"]

        # 各种类服务的服务率范围
        self._service_rate_a_minmax = (config["min_service_rate_a"], config["max_service_rate_a"])
        self._service_rate_q_minmax = (config["min_service_rate_q"], config["max_service_rate_q"])
        self._service_rate_r_minmax = (config["min_service_rate_r"], config["max_service_rate_r"])

        # 传输时延取值范围
        self._tx_u_a_minmax = (config["tx_ua_min"], config["tx_ua_max"])
        self._tx_a_q_minmax = (config["tx_aq_min"], config["tx_aq_max"])
        self._tx_q_r_minmax = (config["tx_qr_min"], config["tx_qr_max"])
        self._tx_r_u_minmax = (config["tx_ru_min"], config["tx_ru_max"])

        # 传输时延矩阵，在initialize_service()里面初始化
        self._tx_u_a = None
        self._tx_a_q = None
        self._tx_q_r = None
        self._tx_r_u = None

        # 各个服务的实例，存储在这里
        # 在在initialize_service()里面初始化
        self._service_A = None  # A类服务，数组
        self._service_q = None  # 同步服务q，全局仅一个
        self._service_R = None  # R类服务，数组；每个元素也是数组，存储子服务。

        # 用于保存用户-服务关联关系的矩阵
        self.user_service_a = np.zeros((self._num_user, self._num_service_a))
        self.user_service_r = np.zeros((self._num_user, self._num_service_r))

        self._assignment_algorithm = config["assignment_algorithm"]

        self._initial_num_server = None     # env初始化之后，各个服务的服务器个数。在save_initial_num_server()中初始化
        self._interaction_delay_without_duplicate = None    # 保存去冗余后的交互时延对。在get_interaction_delay_without_duplicate()中初始化

        self.initialize_transmission_delay()
        self.initialize_user()
        self._services_config = self.initialize_services_config()
        self.assign_users_to_services()
        self.initialize_service()
        self.get_interaction_delay_without_duplicate()
        self.initialize_num_server()

        # print("---- assign_algorithm = {} --------".format(self._assignment_algorithm))
        # print("initial_system_cost = {}".format(self.compute_cost()))
        # print("service_num = {}".format(self._num_service))

        # self.initialize_user()
        # self.initialize_service()
        # self.get_interaction_delay_without_duplicate()
        # self.initialize_num_server()

    """
        初始化各个节点/用户之间的传输时延
    """
    def initialize_transmission_delay(self):
        # 传输时延(用户 --> 逻辑服务器a --> 同步服务器q --> 渲染服务器r --> 用户)，用矩阵保存
        self._tx_u_a = self._rng.integers(self._tx_u_a_minmax[0], self._tx_u_a_minmax[1] + 1, (self._num_user, self._num_service_a))
        self._tx_a_q = self._rng.integers(self._tx_a_q_minmax[0], self._tx_a_q_minmax[1] + 1, self._num_service_a)
        self._tx_q_r = self._rng.integers(self._tx_q_r_minmax[0], self._tx_q_r_minmax[1] + 1, self._num_service_r)
        self._tx_r_u = self._rng.integers(self._tx_r_u_minmax[0], self._tx_r_u_minmax[1] + 1, (self._num_service_r, self._num_user))

    """
        初始化用户实例（不包括与服务的关联关系）
    """
    def initialize_user(self):
        self._instances = []
        self._groups = []
        self._users = []

        instance_id = 0
        group_id = 0
        user_id = 0

        for i in range(self._num_instance):
            instance = Instance(instance_id)
            for g in range(self._num_group_per_instance):
                group = ShareViewGroup(group_id, instance_id)
                for u in range(self._num_user_per_group):
                    user = User(user_id, instance_id, group_id)
                    user._arrival_rate = self._rng.integers(self._min_arrival_rate, self._max_arrival_rate + 1)     # [min_arrival_rate, max_arrival_rate]

                    self._users.append(user)
                    group._users.append(user_id)
                    user_id += 1
                self._groups.append(group)
                instance._groups.append(group_id)
                group_id += 1
            self._instances.append(instance)
            instance_id += 1

    """
        初始化各个服务的配置信息(服务率，价格)，但不真正生成服务的实例。
        用于决定用户与服务的关联关系。
        {
            "service_A": [(50, 2), (40, 1.5), ...],
            "service_Q": ...,
            "service_R": [...]
        }     
    """
    def initialize_services_config(self):
        services_config = {
            "service_A": [],
            "service_Q": None,
            "service_R": []
        }

        # ------------ A服务 ------------
        for _ in range(self._num_service_a):
            service_rate = self._rng.integers(self._service_rate_a_minmax[0], self._service_rate_a_minmax[1] + 1)
            price = (service_rate / self._service_rate_a_minmax[0]) * 2 + 1
            services_config["service_A"].append((service_rate, price))

        # ----------- Q服务 ------------
        service_rate_q = self._rng.integers(self._service_rate_q_minmax[0], self._service_rate_q_minmax[1] + 1)
        price_q = (service_rate_q / self._service_rate_q_minmax[0]) * 2 + 1
        services_config["service_Q"] = (service_rate_q, price_q)

        # ---------- R服务 -------------
        for _ in range(self._num_service_r):
            service_rate = self._rng.integers(self._service_rate_r_minmax[0], self._service_rate_r_minmax[1] + 1)
            price = (service_rate / self._service_rate_r_minmax[0]) * 2 + 1
            services_config["service_R"].append((service_rate, price))

        return services_config

    def assign_users_to_services(self):
        assign_users(self, self._assignment_algorithm)

    """
        初始化服务实例，不包括与用户的关联关系。
        1. 由于关联关系还没决定，所以到达率也还无法计算；对于R服务，子服务也无法初始化
        2. 服务的价格与服务率呈正相关
    """
    def initialize_service(self):
        self._service_A = []
        self._service_R = []        # [[sub-services], [sub-services],...]

        #  ------------------ service_A -----------------------
        for a in range(self._num_service_a):
            service = Service(a, 'a')
            service._service_rate = self._services_config["service_A"][a][0]
            service._price = self._services_config["service_A"][a][1]

            # 计算到达率
            for u in range(self._num_user):
                if self.user_service_a[u][a] == 1:
                    service._arrival_rate += self._users[u]._arrival_rate
                    service._users.append(u)
            self._service_A.append(service)

        # --------------------- service_q --------------------
        # 同步服务器q，仅此一个
        self._service_q = Service(0, 'q')
        self._service_q._service_rate = self._services_config["service_Q"][0]
        self._service_q._price = self._services_config["service_Q"][1]
        for u in range(self._num_user):
            self._service_q._arrival_rate += self._users[u]._arrival_rate
            self._service_q._users.append(u)

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
            service_rate = self._services_config["service_R"][r][0]
            price = self._services_config["service_R"][r][1]

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
        【已弃用】
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
        【已弃用】
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
        【已弃用】
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

        return delay

    """
        获取最大的交互时延
    """
    def get_max_interaction_delay(self):
        interaction_delay_dict = {}
        for key, value in self._interaction_delay_without_duplicate.items():
            user_i = value['users'][0]
            user_j = value['users'][1]
            interaction_delay_dict[(user_i, user_j)] = self.compute_interaction_delay(user_i, user_j)

        max_user_pair = max(interaction_delay_dict, key=lambda k: interaction_delay_dict[k])
        return interaction_delay_dict[max_user_pair], max_user_pair[0], max_user_pair[1]

    """
        获取最大的两个交互时延
    """
    def get_top_two_interaction_delay(self):
        interaction_delay_dict = {}
        for key, value in self._interaction_delay_without_duplicate.items():
            user_i = value["users"][0]
            user_j = value["users"][1]
            interaction_delay_dict[(user_i, user_j)] = self.compute_interaction_delay(user_i, user_j)

        delay_tuple = zip(interaction_delay_dict.values(), interaction_delay_dict.keys())
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
