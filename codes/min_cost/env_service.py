class Service:
    def __init__(self, service_type, node_id=None, user_id=None):

        self.service_type = service_type
        self.user_id = user_id      # 从属哪个用户，仅当服务R时有效
        self.arrival_rate = 0

        """
            以下属性根据所关联的节点、所分配的服务器个数而定
        """
        self.node_id = node_id      # 部署在哪个EdgeNode
        self.service_rate = 0       # 服务率
        self.queuing_delay = 0.
        self.num_server = 0         # 分配的服务器个数（包括num_extra_server）
        self.num_extra_server = 0   # 超出节点容量部分的服务器个数
        self.price = 0.
        self.extra_price = 0.

    def reset(self):
        self.node_id = None     # 部署在哪个EdgeNode
        self.service_rate = 0   # 服务率
        self.queuing_delay = 0.
        self.num_server = 0
        self.num_extra_server = 0
        self.price = 0.
        self.extra_price = 0.

    # 更新服务器的数量，以及排队时延
    def update_num_server(self, n, extra_n, update_queuing_delay=True):
        # if self.service_type == 'A':
        #     print("node_id: {}, num_server: {}".format(self.node_id, self.num_server))

        self.num_server = n
        self.num_extra_server = extra_n if extra_n > 0 else 0
        if update_queuing_delay:
            self.queuing_delay = self.compute_queuing_delay(self.num_server)

    # 初始化服务器个数为刚好达到稳态条件的个数
    # def initialize_num_server(self):
    #     min_num_server = self.get_num_server_for_stability(self.service_rate)
    #     self.update_num_server(min_num_server)

    def get_num_server_for_stability(self, service_rate):
        num_server = 1
        while num_server * service_rate <= self.arrival_rate:
            num_server += 1
        return num_server

    # 增加一台服务器带来的时延减少量
    def reduction_of_delay_when_add_a_server(self):
        num_server = self.num_server + 1
        reduction = self.queuing_delay - self.compute_queuing_delay(num_server)
        return reduction

    # 计算排队时延
    def compute_queuing_delay(self, num_server):
        queuing_delay_iteratively = self.compute_queuing_delay_iteratively(num_server)
        assert queuing_delay_iteratively >= 0.
        return queuing_delay_iteratively

    # 用迭代方法计算排队时延
    def compute_queuing_delay_iteratively(self, num_server):
        lam = float(self.arrival_rate)
        mu = float(self.service_rate)
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