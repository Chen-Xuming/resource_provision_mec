import torch
from torch.utils.data import Dataset
import numpy as np
import itertools
from tqdm import tqdm
from numpy import random

class ServiceDataset(Dataset):
    """
        data_size = 样本数
        seq_len = env的服务个数
    """
    def __init__(self, data_size, env):
        self.env = env
        self.data_size = data_size
        self.seq_len = env._num_service
        self.service_indexes = self.generate_service_index_permutations()
        self.service_list, self.services_infos = self.get_service_list()

    def __len__(self):
        return self.data_size

    """
        返回一组服务的属性
    """
    def __getitem__(self, idx):
        indexes = self.service_indexes[idx]
        attributions = []
        for service_index in indexes:
            attributions.append(self.services_infos[service_index][1])

        item = torch.from_numpy(np.array(attributions)).float()
        return item

    """
        services_list: data_size * seq_len
        返回的是服务的下标，不包含服务的属性
    """
    def generate_service_index_permutations(self):
        services_indexes_list = []
        indexes = [i for i in range(self.seq_len)]      # 服务的下标
        data_iter = tqdm(range(self.data_size), unit='data')
        for i, _ in enumerate(data_iter):
            data_iter.set_description('Data %i/%i' % (i+1, self.data_size))
            services_indexes_list.append(random.permutation(indexes))

        return services_indexes_list

    """
    service_list = 各个服务本身（浅拷贝）
    
    services_infos = 各个服务的信息
        [[('a', 0, none), [0.1, 0.9, 5, 45, 54]],       # attr = [reduction, max_crossing_delay, price, service_rate, arrival_rate]
         [('a', 1, none), [0.1, 1.2, 2, 67, 98]],
         ...]
    """
    def get_service_list(self):
        return self.env.get_services_and_infos()

    """
        1. 将网络输出的点输入到env中(预算不足时停止)，计算max_delay
        2. 重置env
        
        solution = [1, 2, 4, 0, 10, ...]  服务的下标
    """
    def validate_solution(self, solution):
        cost = self.env.compute_cost()
        allocated_server = 0
        for idx in solution:
            if self.service_list[idx]._price + cost > self.env._cost_budget:
                break
            self.service_list[idx].update_num_server(self.service_list[idx]._num_server + 1)
            cost += self.service_list[idx]._price
            allocated_server += 1

        assert cost <= self.env._cost_budget
        max_delay, _, _ = self.env.get_max_interaction_delay()
        self.env.re_initialize_num_server()

        return max_delay, allocated_server

    def get_num_service(self):
        return len(self.service_list)