class EdgeNode:
    def __init__(self, node_id):
        self.node_id = node_id


        # 关联到此节点的服务列表。
        # key = (user_id, service_type), value = Service
        self.service_list = dict()

        self.capacity = 0  # 容量
        self.num_server = 0  # 已分配的服务器数量(包括超出的部分)

        # 超出容量的部分
        # 具体服务A/B/R各有多少个，记录在各个服务中，关联在此节点的服务可以在 self.service_list中找到
        self.num_extra_server = 0

        # 各类服务的单价
        self.price = {
            "A": 0.,
            "B": 0.,
            "R": 0.
        }

        # 超出容量时，服务器的单价
        self.extra_price = {
            "A": 0.,
            "B": 0.,
            "R": 0.
        }

        # 各类服务器的服务率
        self.service_rate = {
            "A": 0,
            "B": 0,
            "R": 0
        }

    def reset(self):
        self.num_server = 0
        self.num_extra_server = 0
        self.service_list.clear()