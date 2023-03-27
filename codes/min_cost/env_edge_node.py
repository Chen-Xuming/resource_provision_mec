class EdgeNode:
    def __init__(self, node_id):
        self.node_id = node_id

        # 关联到此节点的服务列表。
        # key = (user_id, service_type), value = Service
        self.service_list = dict()

        self.capacity = 0  # 容量
        self.num_server = 0  # 已分配的服务器数量

        # 各类服务的单价
        self.price = {
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
