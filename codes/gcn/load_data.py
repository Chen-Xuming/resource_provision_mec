from torch_geometric.data import Data, InMemoryDataset
import torch
import json
import os
from tqdm import tqdm
import copy


class PulpSolutionDataset(InMemoryDataset):
    def __init__(self, transform=None, pre_transform=None):
        super().__init__(None, transform, pre_transform)

        self.max_solution = 0

        # self.raw_data_dir = "../min_max/result/sub_solutions_for_debug"      # debug
        #
        self.raw_data_dir = "../min_max/result/PULP_solutions"
        self.data_list = self.load_data()

        print("Num of Data objects: {}".format(len(self.data_list)))
        print("Max solution: {}".format(self.max_solution))

    def get_num_class(self):
        return self.max_solution + 1

    """
        将数据从json文件中读出，随后为env实例构建一个Data对象。
        返回一个Data列表。
    """
    def load_data(self) -> list:
        data_obj_list = []
        file_list = os.listdir(self.raw_data_dir)
        progress = tqdm(file_list, unit="file")
        for i, file_name in enumerate(file_list):
            data_obj_list.extend(self.deserialization_json_file(self.raw_data_dir + "/" + file_name))
            progress.set_description('Loading Files %i/%i' % (i+1, len(file_list)))
            # break

        return data_obj_list

    """
        解析单个json文件，一个json文件的数据可以构造若干个 Data 对象
    """
    def deserialization_json_file(self, filename) -> list:
        data_objs = []
        raw_data = json.load(open(filename))
        env_config = raw_data["environment_configuration"]

        """
            节点特征
        """
        node_features = []      # [num_node, 5]
        services_info = env_config["services_info"]
        services_dict = dict()                                  # 用于构建 y
        num_service_a = env_config["num_service_a"]
        num_service_r = env_config["num_service_r"]
        for i, service in enumerate(services_info):
            s = service[0]
            s_info = [s[3], s[4], s[5], s[7]]
            node_features.append(s_info)
            services_dict[(s[0], s[1], s[2])] = i

        """
            带权有向邻接矩阵
        """
        transmission_delay = env_config["transmission_delay"]
        edge_index = [[], []]     # [2, num_edge]
        edge_attr = []      # 权重，num_edge 个传输时延 [num_edge, 1]
        service_q_idx = num_service_a   # = 10
        for i in range(num_service_a):
            edge_index[0].append(i)
            edge_index[1].append(service_q_idx)
        for i in range(service_q_idx + 1, len(services_info)):
            edge_index[0].append(service_q_idx)
            edge_index[1].append(i)


        max_u_to_a = [0 for _ in range(num_service_a)]
        max_r_to_u = [0 for _ in range(num_service_r)]
        users = env_config["users_info"]
        for user in users:
            user_id = user[0]
            idx_service_a = user[3][0]
            idx_service_r = user[3][1]
            tx_ua = transmission_delay["tx_u_a"][user_id][idx_service_a]
            tx_ru = transmission_delay["tx_r_u"][idx_service_r][user_id]
            max_u_to_a[idx_service_a] = max(max_u_to_a[idx_service_a], tx_ua)
            max_r_to_u[idx_service_r] = max(max_r_to_u[idx_service_r], tx_ru)

        for i in range(num_service_a):
            edge_attr.append(transmission_delay["tx_a_q"][i] + max_u_to_a[i])
        for i in range(service_q_idx + 1, len(services_info)):
            idx_service_r = services_info[i][0][0]
            edge_attr.append(transmission_delay["tx_q_r"][idx_service_r] + max_r_to_u[idx_service_r])

        """
            各个节点的标签（添加的服务器个数）
        """
        budget = 50
        max_budget = 230
        while budget <= max_budget:
            solution = raw_data[str(budget)]["solution"]

            # 把budget插入到节点特征里面
            nf = copy.deepcopy(node_features)
            for node_feat in nf:
                node_feat.append(budget)

            label = [0 for _ in range(len(nf))]
            for sol in solution:
                service_type = sol[0][0]
                service_id = sol[0][1]
                num_server = sol[1]

                if num_server > self.max_solution:
                    self.max_solution = num_server

                if service_type == "a":
                    label[service_id] = num_server
                elif service_type == "q":
                    label[num_service_a] = num_server
                elif service_type == "r":
                    idx = services_dict[(service_id[0], service_id[1], service_type)]
                    label[idx] = int(num_server)

            # 创建Data对象
            graph = Data(x=torch.tensor(nf, dtype=torch.float32),
                         edge_index=torch.tensor(edge_index, dtype=torch.long),
                         edge_attr=torch.Tensor(edge_attr),
                         y=torch.Tensor(label))
            data_objs.append(graph)

            # if budget == 50:
            #     print("--------- node_features -------")
            #     print(nf)
            #     print("--------- edge_index -----------")
            #     print(edge_index)
            #     print("--------- edge_attr ------------")
            #     print(edge_attr)
            #     print("--------  label -----------------")
            #     print(label)
            #     print(graph)

            budget += 20

        return data_objs


    def len(self) -> int:
        return len(self.data_list)

    def get(self, idx: int) -> Data:
        return self.data_list[idx]


if __name__ == '__main__':
    dataset = PulpSolutionDataset()
