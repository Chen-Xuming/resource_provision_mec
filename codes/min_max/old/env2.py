import numpy as np
import math

"""
1. 传输时延 = [4, 6] + [2, 4] + [2, 4] + 0 + [2, 4] + [4, 6] = [14, 24]ms

2. 处理时延 = [5, 10] + [2, 5] + [2, 5] + [4, 10] + [20, 25] = [33, 55]ms
   服务率:
   A = [100, 200]
   Q = [200, 500]
   B = [200, 500]
   C = [100, 250]
   R = [40, 50]

传输时延 + 处理时延 = [47, 79]

"""


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

"""
    增加了两个细分服务：
    1. C服务的子服务
    2. R服务的子服务
"""
class User:
    def __init__(self, id, instance_id, group_id):
        self._id = id
        self._instance_id = instance_id
        self._group_id = group_id
        self._arrival_rate = 0
        self._service_a = 0
        self._service_b = 0

        self._service_c = 0
        self._sub_service_c = 0     # C服务中的子服务

        self._service_r = 0
        self._sub_service_r = 0     # R服务中的子服务

    def __str__(self):
        return 'instance: {} group: {} user: {} arrival_rate:{} service a:{} b:{} c:{} sub_c:{} r:{} sub_r:{}'.format(
                self._instance_id,
                self._group_id,
                self._id, self._arrival_rate,
                self._service_a,
                self._service_b,
                self._service_c,
                self._sub_service_c,
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

        # todo: 可能不需要了？
        self._group = set()

        self._price = 0
        self._num_server = 1
        self._queuing_delay = 0.

        # 当仅时C或R类服务时有效
        self._sub_id = None


    def __str__(self):
        return 'service: {} {},{} service_rate: {} arrival_rate: {} users:{} price:{} num_server:{} arr/ser = {} queuing_delay:{} delay_reduction: {}'.format(
            self._type,
            self._id,
            self._sub_id,
            self._service_rate,
            self._arrival_rate,
            self._users,
            # self._group,
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
                # self._users,
                # self._group,
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
        num_server = self._num_server   # 初始化是1
        while num_server * self._service_rate <= self._arrival_rate:
            num_server += 1
        self.update_num_server(num_server)

        # num_server = round(self._arrival_rate / self._service_rate) + 1
        # self.update_num_server(num_server)

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

    def compute_queuing_delay_2(self, num_server):
        import math
        lam = float(self._arrival_rate)
        mu = float(self._service_rate)
        c = num_server
        r = lam / mu
        rho = r / c
        # print('type {}:lam {} mu {} c {} r {} rho {}'.format(self._type,lam,mu,c,r,rho))
        assert rho < 1
        p0 = 1 / (math.pow(r, c) / (float(math.factorial(c)) * (1 - rho)) + sum(
            [math.pow(r, n) / float(math.factorial(n)) for n in range(0, c)]))
        queuing_delay = (math.pow(r, c) / (float(math.factorial(c)) * float(c) * mu * math.pow(1 - rho, 2))) * p0
        return queuing_delay

    # 计算排队时延
    def compute_queuing_delay(self, num_server):
        queuing_delay_iteratively = self.compute_queuing_delay_iteratively(num_server)
        # print(queuing_delay_iteratively)
        assert queuing_delay_iteratively >= 0.
        return queuing_delay_iteratively

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
        "num_service_b": num_service_b,
        "num_service_c": num_service_c,
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
        "min_tx": min_tx,       # 传输时延
        "max_tx": max_tx,
    }
"""
class Env:
    def __init__(self, config, rng, seed_sequence):
        self._seed_sequence = seed_sequence
        self._rng = rng
        self._num_instance = config["num_instance"]
        self._num_service_a = config["num_service_a"]
        self._num_service_b = config["num_service_b"]
        self._num_service_c = config["num_service_c"]
        self._num_service_r = config["num_service_r"]

        # todo: 引入了子服务之后，service的数量与group有关
        self._num_service = 0

        self._num_group_per_instance = config["num_group_per_instance"]  # 每个instance的组数
        self._num_user_per_group = config["num_user_per_group"]  # 每个组的用户数

        self._max_arrival_rate = config["max_arrival_rate"]  # 61
        self._min_arrival_rate = config["min_arrival_rate"]

        self._max_service_rate = config["max_service_rate"]  # 120
        self._min_service_rate = config["min_service_rate"]  # 100

        self._max_price = config["max_price"]
        self._min_price = config["min_price"]

        self._trigger_probability = config["trigger_probability"]

        self._max_tx = config["max_tx"]
        self._min_tx = config["min_tx"]

        self._cost_budget = 0       # 初始化开销 + 预算
        self._budget_addition = config["budget_addition"]

        self.initialize_user()
        self.correct_initialize_user()
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

        # todo:
        self._num_user = self._num_instance * self._num_group_per_instance * self._num_user_per_group
        self.user_service_a = np.zeros((self._num_user, self._num_service_a))
        self.user_service_b = np.zeros((self._num_user, self._num_service_b))
        self.user_service_c = np.zeros((self._num_user, self._num_service_c))
        self.user_service_r = np.zeros((self._num_user, self._num_service_r))

        for i in range(self._num_instance):
            instance = Instance(instance_id)
            for g in range(self._num_group_per_instance):
                group = ShareViewGroup(group_id, instance_id)
                for u in range(self._num_user_per_group):
                    user = User(user_id, instance_id, group_id)
                    user._arrival_rate = self._rng.integers(self._min_arrival_rate, self._max_arrival_rate + 1)     # [min_arrival_rate, max_arrival_rate]

                    user._service_a = self._rng.integers(0, self._num_service_a)

                    # 注意：B和C是成套出现的，即如果选了b_k，那么也应该关联c_k
                    b_c = self._rng.integers(0, self._num_service_b)
                    user._service_b = b_c
                    user._service_c = b_c

                    user._service_r = self._rng.integers(0, self._num_service_r)

                    # todo: ------------------ 子服务 --------------------
                    # 用户的子服务分配在 initialize_service() 部分

                    # 二维矩阵保存关联关系
                    self.user_service_a[user._id][user._service_a] = 1
                    self.user_service_b[user._id][user._service_b] = 1
                    self.user_service_c[user._id][user._service_c] = 1
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

        # 传输时延
        self._tx_u_a = self._rng.integers(4, 6 + 1, (self._num_user, self._num_service_a))
        self._tx_a_b0 = self._rng.integers(2, 4 + 1, self._num_service_a)
        self._tx_b0_b = self._rng.integers(2, 4 + 1, self._num_service_b)
        self._tx_b_c = self._rng.integers(0, 0 + 1, (self._num_service_b, self._num_service_c))
        self._tx_c_r = self._rng.integers(2, 4 + 1, (self._num_service_c, self._num_service_r))
        self._tx_r_u = self._rng.integers(4, 6 + 1, (self._num_service_r, self._num_user))

        self._service_A = []
        self._service_B = []
        self._service_C = []        # [[sub-services], [sub-services],...]
        self._service_R = []        # [[sub-services], [sub-services],...]

        # service_A
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


        # service_b0
        # b0是文章里的master state service p
        self._service_b0 = Service(0, 'b0')
        # self._service_b0._service_rate = self._rng.integers(self._min_service_rate, self._max_service_rate + 1)
        self._service_b0._service_rate = self._rng.integers(200, 500 + 1)
        for u in range(self._num_user):
            self._service_b0._arrival_rate += self._users[u]._arrival_rate
            self._service_b0._users.append(u)

        self._service_b0._price = self._rng.integers(self._min_price, self._max_price + 1)
        # self._service_b0._price = self._service_b0._service_rate / self._min_service_rate * 2 + 1

        # service_B
        for b in range(self._num_service_b):
            service = Service(b, 'b')
            # service._service_rate = self._rng.integers(self._min_service_rate, self._max_service_rate + 1)
            service._service_rate = self._rng.integers(200, 500 + 1)
            service._arrival_rate = self._service_b0._arrival_rate
            for u in range(self._num_user):
                if self.user_service_b[u][b] == 1:
                    service._users.append(u)

            service._price = self._rng.integers(self._min_price, self._max_price + 1)
            # service._price = service._service_rate / self._min_service_rate * 2 + 1

            self._service_B.append(service)

        # service_C
        for c in range(self._num_service_c):
            # ----找出站点c上有哪些用户组-----
            sub_service_list = []
            share_group = {}
            for u in range(self._num_user):
                if self.user_service_c[u][c] == 1:
                    if self._users[u]._group_id not in share_group:
                        share_group[self._users[u]._group_id] = [u]
                    else:
                        share_group[self._users[u]._group_id].append(u)

            # 同一个站点的每个子服务的服务器服务率、价格是一样的
            price = self._rng.integers(self._min_price, self._max_price + 1)
            # service_rate = self._rng.integers(self._min_service_rate, self._max_service_rate + 1)
            service_rate = self._rng.integers(100, 250 + 1)

            # ----- 为每一个用户组新建一个子服务 ----------
            sub_service_id = 0
            for group, users in share_group.items():
                sub_service = Service(c, 'c')
                sub_service._sub_id = sub_service_id
                sub_service._users = users
                sub_service._service_rate = service_rate
                sub_service._price = price

                # todo
                sub_service._arrival_rate = self._service_b0._arrival_rate * self._trigger_probability

                for user_id in users:
                    self._users[user_id]._sub_service_c = sub_service_id
                    # sub_service._arrival_rate += self._users[user]._arrival_rate

                sub_service_list.append(sub_service)
                sub_service_id += 1
            self._service_C.append(sub_service_list)

        # service_R
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

                # todo
                sub_service._arrival_rate = self._service_b0._arrival_rate * self._trigger_probability

                for user_id in users:
                    self._users[user_id]._sub_service_r = sub_service_id

                sub_service_list.append(sub_service)
                sub_service_id += 1
            self._service_R.append(sub_service_list)

        num_service_c = 0
        num_service_r = 0
        for sub_service_list in self._service_C:
            num_service_c += len(sub_service_list)
        for sub_service_list in self._service_R:
            num_service_r += len(sub_service_list)
        self._num_service = self._num_service_a + 1 + self._num_service_b + num_service_c + num_service_r

    # 避免同一用户组的任意两个用户，关联的服务c不同，而关联的服务r相同
    # 如果出现此种情况，将用户关联相同的服务c
    # 此外，由于B和C是一一对应，那么他们的B服务也相同
    def correct_initialize_user(self):
        for group in self._groups:
            for u in group._users:
                u_cmp = u + 1
                while u_cmp <= group._users[-1]:
                    if self._users[u]._service_c != self._users[u_cmp]._service_c and self._users[u]._service_r == self._users[u_cmp]._service_r:
                        # 改变service_c关联
                        self.user_service_c[u_cmp][self._users[u_cmp]._service_c] = 0
                        self._users[u_cmp]._service_c = self._users[u]._service_c
                        self.user_service_c[u_cmp][self._users[u_cmp]._service_c] = 1

                        # 改变service_b关联
                        self.user_service_b[u_cmp][self._users[u_cmp]._service_b] = 0
                        self._users[u_cmp]._service_b = self._users[u]._service_b
                        self.user_service_b[u_cmp][self._users[u_cmp]._service_b] = 1

                    u_cmp += 1

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
                a, b0, b, c, sub_c, r, sub_r = self.get_service_index(i, j)
                # service_set = {a, b0, b, (c, sub_c), (r, sub_r)}

                service_set = [a, b0, b, (c, sub_c), (r, sub_r)]    # 这里各个编号可能会相同，所以不能用集合

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
                a, b0, b, c, sub_c, r, sub_r = self.get_service_index(i, j)
                service_set = [a, b0, b, (c, sub_c), (r, sub_r)]

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

    def get_delay_index(self, user_i , user_j):
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
        b0 = 0  # b0是文章里的master state service p
        b = self._users[user_j]._service_b

        c = self._users[user_j]._service_c
        sub_c = self._users[user_j]._sub_service_c

        r = self._users[user_j]._service_r
        sub_r = self._users[user_j]._sub_service_r

        return a, b0, b, c, sub_c, r, sub_r

    # 初始化服务器个数
    def initialize_num_server(self):
        for i in range(self._num_service_a):
            self._service_A[i].initialize_num_server()

        self._service_b0.initialize_num_server()

        for i in range(self._num_service_b):
            self._service_B[i].initialize_num_server()

        for i in range(self._num_service_c):
            # 对一个C站点，为其每一个子服务初始化服务器
            for sub_service in self._service_C[i]:
                sub_service.initialize_num_server()

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

    def re_initialize_num_server(self):
        for i, service in enumerate(self._service_A):
            service.update_num_server(self._initial_num_server["service_a"][i])

        self._service_b0.update_num_server(self._initial_num_server["service_b0"])

        for i, service in enumerate(self._service_B):
            service.update_num_server(self._initial_num_server["service_b"][i])

        for i in range(self._num_service_c):
            for j, service in enumerate(self._service_C[i]):
                service.update_num_server(self._initial_num_server["service_c"][i][j])

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

        cost += self._service_b0._num_server * self._service_b0._price

        for service in self._service_B:
            cost += service._num_server * service._price

        # C服务开销：所有子服务开销的总和
        for i in range(self._num_service_c):
            for sub_service in self._service_C[i]:
                cost += sub_service._num_server * sub_service._price

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
            "service_b0": 0,
            "service_b": [],
            "service_c": [],    # [[], [], ...]
            "service_r": []
        }

        for service in self._service_A:
            self._initial_num_server["service_a"].append(service._num_server)

        self._initial_num_server["service_b0"] = self._service_b0._num_server

        for service in self._service_B:
            self._initial_num_server["service_b"].append(service._num_server)

        for i in range(self._num_service_c):
            sub_service_num_servers = []
            for service in self._service_C[i]:
                sub_service_num_servers.append(service._num_server)
            self._initial_num_server["service_c"].append(sub_service_num_servers)

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
                        self._service_b0._queuing_delay + \
                        self._service_B[self._users[user_j]._service_b]._queuing_delay + \
                        self._service_C[self._users[user_j]._service_c][self._users[user_j]._sub_service_c]._queuing_delay + \
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
                             self._tx_a_b0[self._users[user_i]._service_a] + \
                             self._tx_b0_b[self._users[user_j]._service_b] + \
                             self._tx_b_c[self._users[user_j]._service_b][self._users[user_j]._service_c] + \
                             self._tx_c_r[self._users[user_j]._service_c][self._users[user_j]._service_r] + \
                             self._tx_r_u[self._users[user_j]._service_r][user_j]

        # 排队时延(s)
        queuing_delay = self.compute_queuing_delay(user_i, user_j)

        # 处理时延(s)
        processing_time = 1 / self._service_A[self._users[user_i]._service_a]._service_rate +\
                            1 / self._service_b0._service_rate +\
                            1 / self._service_B[self._users[user_j]._service_b]._service_rate +\
                            1 / self._service_C[self._users[user_j]._service_c][self._users[user_j]._sub_service_c]._service_rate +\
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

    """
        =============================================
        未重写函数
        
        Env:
            save_price_list
            get_service_type_and_index  
            get_feature      
            get_topk_queuing_delay_in_interaction_delay
            get_max_queuing_delay_in_interaction_delay_old
            get_total_queuing_delay
            valid_action_mask
            observe
            act
            get_feature_without_duplicate
            get_topk_feature
    """

    """
        debug
    """
    def check_users(self):
        instance = self._instances[0]
        for group_id in instance._groups:
            print("---------- group: {} ------------".format(group_id))
            for user_id in self._groups[group_id]._users:
                user = self._users[user_id]
                print("user: {}, arrival_rate: {}, services: {}".format(user_id, user._arrival_rate, (user._service_a, 0, user._service_b, (user._service_c, user._sub_service_c),
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

        print("\n---------- service b0 -------------")
        service_b0 = self._service_b0
        print("service: ({}, {}), service_rate: {}, arrival_rate: {}, users: {}, price: {}, num_server: {}, queuinng_delay: {}"
              .format(service_b0._type,
                    service_b0._id,
                    service_b0._service_rate,
                    service_b0._arrival_rate,
                    service_b0._users,
                    service_b0._price,
                    service_b0._num_server,
                    service_b0._queuing_delay))

        print("\n---------- service B -------------")
        for service in self._service_B:
            print("service: ({}, {}), service_rate: {}, arrival_rate: {}, users: {}, price: {}, num_server: {}, queuinng_delay: {}"
                  .format(service._type,
                          service._id,
                          service._service_rate,
                          service._arrival_rate,
                          service._users,
                          service._price,
                          service._num_server,
                          service._queuing_delay))

        print("\n---------- service C -------------")
        for i in range(self._num_service_c):
            for service in self._service_C[i]:
                print("service: ({}, {}), service_rate: {}, arrival_rate: {}, users: {}, price: {}, num_server: {}, queuinng_delay: {}"
                      .format(service._type,
                              (service._id, service._sub_id),
                              service._service_rate,
                              service._arrival_rate,
                              service._users,
                              service._price,
                              service._num_server,
                              service._queuing_delay))

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



















