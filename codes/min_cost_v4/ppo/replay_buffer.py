import numpy
import torch
import numpy as np

"""
    不定长缓存池
"""
class ReplayBuffer:
    def __init__(self, args):
        self.s = []
        self.a = []
        self.a_logprob = []
        self.r = []
        self.s_ = []
        self.dw = []
        self.done = []
        self.mask = []
        self.full_mask = []

        self.count = 0

    def store(self, s, a, a_logprob, r, s_, dw, done, mask, full_mask):
        self.s.append(s)
        self.a.append(a)
        self.a_logprob.append(a_logprob)
        self.r.append(r)
        self.s_.append(s_)
        self.dw.append(dw)
        self.done.append(done)
        self.mask.append(mask)
        self.full_mask.append(full_mask)

        self.count += 1

    def numpy_to_tensor(self):
        s = torch.tensor(np.array(self.s), dtype=torch.float)
        a = torch.tensor(np.array(self.a).reshape((self.count, 1)), dtype=torch.long)                       # (batch_size, ) --> (batch_size, 1)
        a_logprob = torch.tensor(np.array(self.a_logprob).reshape((self.count, 1)), dtype=torch.float)      # (batch_size, ) --> (batch_size, 1)
        r = torch.tensor(np.array(self.r).reshape((self.count, 1)), dtype=torch.float)                      # (batch_size, ) --> (batch_size, 1)
        s_ = torch.tensor(np.array(self.s_), dtype=torch.float)
        dw = torch.tensor(np.array(self.dw).reshape((self.count, 1)), dtype=torch.float)                    # (batch_size, ) --> (batch_size, 1)
        done = torch.tensor(np.array(self.done).reshape((self.count, 1)), dtype=torch.float)                # (batch_size, ) --> (batch_size, 1)
        mask = torch.tensor(np.array(self.mask), dtype=torch.bool)
        full_mask = torch.tensor(np.array(self.full_mask), dtype=torch.bool)

        return s, a, a_logprob, r, s_, dw, done, mask, full_mask

    def clear_all(self):
        self.s.clear()
        self.a.clear()
        self.a_logprob.clear()
        self.r.clear()
        self.s_.clear()
        self.dw.clear()
        self.done.clear()
        self.mask.clear()
        self.full_mask.clear()

        self.count = 0


# class ReplayBuffer:
#     def __init__(self, args):
#         self.s = np.zeros((args.batch_size, args.num_edge_node, args.node_state_dim))        # 每个state是二维矩阵
#         self.a = np.zeros((args.batch_size, 1))
#         self.a_logprob = np.zeros((args.batch_size, 1))
#         self.r = np.zeros((args.batch_size, 1))
#         self.s_ = np.zeros((args.batch_size, args.num_edge_node, args.node_state_dim))
#         self.dw = np.zeros((args.batch_size, 1))
#         self.done = np.zeros((args.batch_size, 1))
#
#         self.mask = np.zeros((args.batch_size, args.action_dim * 2))     # mask
#         self.full_mask = np.zeros((args.batch_size, args.action_dim ** 2))
#
#         self.count = 0
#
#     def store(self, s, a, a_logprob, r, s_, dw, done, mask, full_mask):
#         self.s[self.count] = s
#         self.a[self.count] = a
#         self.a_logprob[self.count] = a_logprob
#         self.r[self.count] = r
#         self.s_[self.count] = s_
#         self.dw[self.count] = dw
#         self.done[self.count] = done
#         self.mask[self.count] = mask
#         self.full_mask[self.count] = full_mask
#
#         self.count += 1
#
#     def numpy_to_tensor(self):
#         s = torch.tensor(self.s, dtype=torch.float)
#         a = torch.tensor(self.a, dtype=torch.long)  # In discrete action space, 'a' needs to be torch.long
#         a_logprob = torch.tensor(self.a_logprob, dtype=torch.float)
#         r = torch.tensor(self.r, dtype=torch.float)
#         s_ = torch.tensor(self.s_, dtype=torch.float)
#         dw = torch.tensor(self.dw, dtype=torch.float)
#         done = torch.tensor(self.done, dtype=torch.float)
#         mask = torch.tensor(self.mask, dtype=torch.bool)
#         full_mask = torch.tensor(self.full_mask, dtype=torch.bool)
#
#         return s, a, a_logprob, r, s_, dw, done, mask, full_mask