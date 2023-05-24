import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.data.batch import Batch


class ReplayBuffer:
    def __init__(self, args):
        """
            每个 state 包括图的特征和用户特征。
            1. graph_state： 一个 Data 对象，记录图的信息，包括节点特征、邻接关系、边的权值
            2. user_state：一个向量
        """

        self.graph_state = []
        self.user_state = []

        self.graph_state_ = []
        self.user_state_ = []       # 实际上user_state并不会发生变化

        self.a = np.zeros((args.batch_size, 2))
        self.a_logprob = np.zeros((args.batch_size, 1))
        self.r = np.zeros((args.batch_size, 1))
        self.dw = np.zeros((args.batch_size, 1))
        self.done = np.zeros((args.batch_size, 1))
        self.count = 0

    """
        g 和 g_ 是 Data 对象
    """
    def store(self, g, g_, u, a, a_logprob, r, dw, done):
        self.graph_state.append(g)
        self.graph_state_.append(g_)
        self.user_state.append(u)
        self.user_state_.append(u)

        self.a[self.count] = a
        self.a_logprob[self.count] = a_logprob
        self.r[self.count] = r
        self.dw[self.count] = dw
        self.done[self.count] = done
        self.count += 1

    """
        1. 将 Data 列表直接返回
        2. 其余的 numpy/list 转 Tensor
    """
    def transform(self):
        # batch = Batch.from_data_list(self.graph_state)
        # batch_ = Batch.from_data_list(self.graph_state_)

        u = torch.tensor(np.array(self.user_state), dtype=torch.float)
        u_ = torch.tensor(np.array(self.user_state_), dtype=torch.float)

        a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
        a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
        r = torch.tensor(self.r, dtype=torch.float)
        dw = torch.tensor(self.dw, dtype=torch.float)
        done = torch.tensor(self.done, dtype=torch.float)

        return self.graph_state, self.graph_state_, u, u_, a, a_logprob, r, dw, done

    def clear(self):
        self.count = 0
        self.graph_state.clear()
        self.graph_state_.clear()
        self.user_state.clear()
        self.user_state_.clear()
