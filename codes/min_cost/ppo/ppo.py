import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical
from torch_geometric.nn import GCNConv
from torch_geometric.nn import GeneralConv

from torch_geometric.utils import unbatch
from torch_geometric.data.batch import Batch


# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    """
        env 用于获取关于网络拓扑的信息（供GCN使用）
    """
    def __init__(self, args, env):
        super(Actor, self).__init__()

        self.env = env

        # self.gcn = GCNConv(env.node_state_dim, args.node_embedding_size)     # 学习节点特征

        self.gcn = GeneralConv(env.node_state_dim, args.node_embedding_size, 2)
        self.gcn2 = GeneralConv(env.node_state_dim, args.node_embedding_size, 2)
        self.fc1 = nn.Linear(env.user_state_dim, args.user_embedding_size)   # 学习用户特征
        self.fc2 = nn.Linear(env.num_edge_node * args.node_embedding_size + args.user_embedding_size, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.hidden_dim // 2)
        self.fc4 = nn.Linear(args.hidden_dim // 2, env.num_edge_node)

        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            # orthogonal_init(self.gcn)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4, gain=0.01)

    """
        如果is_batch == True, graph是一个Batch对象，否则是一个Data对象
    """
    def forward(self, graph, user_s, is_batch=False):
        # print("[actor.forward]")

        if is_batch:
            # print("[actor.forward] is_batch = True")

            # 将Data list合成Batch
            batch = Batch.from_data_list(graph)

            node_state = self.activate_func(
                self.gcn(batch.x, batch.edge_index, edge_attr=batch.edge_attr))  # batch_size * num_node * node_feat_dim
            node_state = F.dropout(node_state, p=0.5)
            node_state = self.activate_func(self.gcn2(batch.x, batch.edge_index, edge_attr=batch.edge_attr))
            node_state = F.dropout(node_state, p=0.5)

            user_state = self.activate_func(self.fc1(user_s))

            # 将batch拆开，得到batch_size个子图，
            # 将各个子图的节点特征拼接成一维，然后又和用户特征拼接
            # 随后又将 batch_size 个一维特征向量堆叠在一起。
            split_node_state = unbatch(node_state, batch=batch.batch)
            states = []
            for i, graph in enumerate(split_node_state):
                states.append(torch.cat([graph.view(-1), user_state[i]]))
            states = torch.stack(states)

            states = self.activate_func(self.fc2(states))
            states = self.activate_func(self.fc3(states))
            states = self.fc4(states)

            states = torch.unsqueeze(states, dim=0)
            a_prob = torch.softmax(states, dim=1)
            return a_prob

        else:
            node_state = self.activate_func(self.gcn(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr))
            node_state = F.dropout(node_state, p=0.5)
            node_state = self.activate_func(self.gcn2(x=graph.x, edge_index=graph.edge_index, edge_attr=graph.edge_attr))
            node_state = F.dropout(node_state, p=0.5)

            user_state = torch.tensor(user_s, dtype=torch.float)
            user_state = self.activate_func(self.fc1(user_state))

            state = torch.cat([node_state.view(-1), user_state], dim=0)
            state = self.activate_func(self.fc2(state))
            state = self.activate_func(self.fc3(state))
            state = self.fc4(state)

            state = torch.unsqueeze(state, dim=0)
            a_prob = torch.softmax(state, dim=1)
            return a_prob


class Critic(nn.Module):
    """
        env 用于获取关于网络拓扑的信息（供GCN使用）
    """
    def __init__(self, args, env):
        super(Critic, self).__init__()

        self.env = env

        # self.gcn = GCNConv(env.node_state_dim, args.node_embedding_size)

        self.gcn = GeneralConv(env.node_state_dim, args.node_embedding_size, 2)
        self.gcn2 = GeneralConv(env.node_state_dim, args.node_embedding_size, 2)

        self.fc1 = nn.Linear(env.user_state_dim, args.user_embedding_size)
        self.fc2 = nn.Linear(env.num_edge_node * args.node_embedding_size + args.user_embedding_size, args.hidden_dim)
        self.fc3 = nn.Linear(args.hidden_dim, args.hidden_dim // 2)
        self.fc4 = nn.Linear(args.hidden_dim // 2, 1)

        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            print("------use_orthogonal_init------")
            # orthogonal_init(self.gcn)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            orthogonal_init(self.fc3)
            orthogonal_init(self.fc4)

    """
        如果is_batch == True, graph是一个Data 列表，否则是一个Data对象
    """
    def forward(self, graph, user_s, is_batch=False):
        # print("[critic.forward]")
        if is_batch:
            # print("[critic.forward] is_batch = True")

            # 将Data list合成Batch
            batch = Batch.from_data_list(graph)

            node_state = self.activate_func(self.gcn(batch.x, batch.edge_index, edge_attr=batch.edge_attr))   # batch_size * num_node * node_feat_dim
            node_state = F.dropout(node_state, p=0.5)
            node_state = self.activate_func(self.gcn2(batch.x, batch.edge_index, edge_attr=batch.edge_attr))
            node_state = F.dropout(node_state, p=0.5)

            user_state = self.activate_func(self.fc1(user_s))

            # 将batch拆开，得到batch_size个子图，
            # 将各个子图的节点特征拼接成一维，然后又和用户特征拼接
            # 随后又将 batch_size 个一维特征向量堆叠在一起。
            split_node_state = unbatch(node_state, batch=batch.batch)
            states = []
            for i, graph in enumerate(split_node_state):
                states.append(torch.cat([graph.view(-1), user_state[i]]))
            states = torch.stack(states)

            states = self.activate_func(self.fc2(states))
            states = self.activate_func(self.fc3(states))
            v_s = self.fc4(states)
            return v_s

        else:
            node_state = self.activate_func(self.gcn(graph.x, graph.edge_index, edge_attr=graph.edge_attr))
            node_state = F.dropout(node_state, p=0.5)
            node_state = self.activate_func(self.gcn2(graph.x, graph.edge_index, edge_attr=graph.edge_attr))
            node_state = F.dropout(node_state, p=0.5)

            user_state = torch.tensor(user_s, dtype=torch.float)
            user_state = self.activate_func(self.fc1(user_state))

            state = torch.cat([node_state.view(-1), user_state], dim=0)
            state = self.activate_func(self.fc2(state))
            state = self.activate_func(self.fc3(state))

            v_s = self.fc4(state)
            return v_s

        # node_state = torch.tensor(s[0], dtype=torch.float32)  # N * num_node_feature
        # user_state = torch.tensor(s[1], dtype=torch.float32)  # 1 * num_user_feature
        #
        # node_state = self.activate_func(self.gcn(node_state, self.env.edge_index, self.env.edge_weight))
        # user_state = self.activate_func(self.fc1(user_state))
        # node_state = node_state.view(-1)
        # state = torch.cat([node_state, user_state], dim=0)
        # state = self.activate_func(self.fc2(state))
        #
        # v_s = self.fc3(state)
        # return v_s


class PPO:
    def __init__(self, args, env):

        self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size
        self.max_train_steps = args.max_train_steps
        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.actor = Actor(args, env)
        self.critic = Critic(args, env)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    """
        evaluate的时候，选择最高概率的作为action
        s 是 numpy array, actor.forward中自行转换成tensor.
    """
    # TODO: 如果想要避免违背时延约束的情况，可以增加一 mask layer参数来屏蔽非法action，
    # TODO: 也可以像 choose_action 那样，使用 Categorical() 来采样，增加输出的多样性。

    def evaluate(self, graph_s, user_s):
        a_prob = self.actor(graph_s, user_s, is_batch=False).detach().numpy().flatten()
        a = np.argmax(a_prob)
        return a

    def choose_action(self, graph_state, user_state):
        with torch.no_grad():
            dist = Categorical(probs=self.actor(graph_state, user_state, is_batch=False))
            a = dist.sample()
            a_logprob = dist.log_prob(a)
        return a.numpy()[0], a_logprob.numpy()[0]

    def update(self, replay_buffer, total_steps):
        Data_list, Data_list_, u, u_, a, a_logprob, r, dw, done = replay_buffer.transform()     # Get training data

        """
            Calculate the advantage using GAE(General Advantage Estimation)
            'dw=True' means dead or win, there is no next state s'
            'done=True' represents the terminal of an episode(dead or win or reaching the max_episode_steps). When calculating the adv, if done=True, gae=0
        
            dw = 选到的节点不满足时延约束
            done = 不满足时延约束 or 全部用户处理完毕
        """
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(Data_list, u, is_batch=True)
            vs_ = self.critic(Data_list_, u_, is_batch=True)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs
            for delta, d in zip(reversed(deltas.flatten().numpy()), reversed(done.flatten().numpy())):
                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)
            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        # Optimize policy for K epochs:
        for _ in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
                # # 取出Batch中index对应的那些Data对象
                # # 然后把它们组成新的Batch
                # data_objs = batch.to_data_list()
                # mini_batch = Batch.from_data_list(data_objs[index])

                # 获取index对应的Data，放在一个list里面
                data_mini_batch = []
                for idx in index:
                    data_mini_batch.append(Data_list[idx])

                dist_now = Categorical(probs=self.actor(data_mini_batch, u[index], is_batch=True))
                dist_entropy = dist_now.entropy().view(-1, 1)  # shape(mini_batch_size X 1)
                a_logprob_now = dist_now.log_prob(a[index].squeeze()).view(-1, 1)  # shape(mini_batch_size X 1)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)
                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()

                v_s = self.critic(data_mini_batch, u[index], is_batch=True)
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        if self.use_lr_decay:  # Trick 6:learning rate Decay
            self.lr_decay(total_steps)


    def lr_decay(self, total_steps):
        lr_a_now = self.lr_a * (1 - total_steps / self.max_train_steps)
        lr_c_now = self.lr_c * (1 - total_steps / self.max_train_steps)
        for p in self.optimizer_actor.param_groups:
            p['lr'] = lr_a_now
        for p in self.optimizer_critic.param_groups:
            p['lr'] = lr_c_now

        print("step: {} / {}, lr = {}".format(total_steps, self.max_train_steps, lr_a_now))




