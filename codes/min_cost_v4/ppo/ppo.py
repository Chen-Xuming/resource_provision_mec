import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from torch.distributions import Categorical

# import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"



# 使用GPU
device = torch.device('cpu')
if torch.cuda.is_available():
    device = torch.device('cuda:1')
    torch.cuda.empty_cache()
    print("[Use Device]: " + str(torch.cuda.get_device_name(device)))
else:
    print("[Use Device]: CPU")




# Trick 8: orthogonal initialization
def orthogonal_init(layer, gain=1.0):
    nn.init.orthogonal_(layer.weight, gain=gain)
    nn.init.constant_(layer.bias, 0)

class Actor(nn.Module):
    def __init__(self, args):
        super(Actor, self).__init__()

        # # ------------------- 25 节点 -----------------------------------------
        # self.conv1 = torch.nn.Conv1d(args.node_state_dim, 64, kernel_size=1)
        # self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=1)
        # self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=1)
        #
        # self.fc1 = torch.nn.Linear(256, 128)
        # self.fc2 = torch.nn.Linear(128, 64)
        # self.fc3 = torch.nn.Linear(64, args.num_edge_node * 2)
        #
        # self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        #
        # if args.use_orthogonal_init:
        #     orthogonal_init(self.conv1)
        #     orthogonal_init(self.conv2)
        #     orthogonal_init(self.conv3)
        #     orthogonal_init(self.fc1)
        #     orthogonal_init(self.fc2)
        #     orthogonal_init(self.fc3)

        # ------------------- 40 节点 -----------------------------------------
        # self.conv1 = torch.nn.Conv1d(args.node_state_dim, 64, kernel_size=1)
        # self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=1)
        # self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=1)
        # self.conv4 = torch.nn.Conv1d(256, 512, kernel_size=1)
        #
        # self.fc1 = torch.nn.Linear(512, 256)
        # self.fc2 = torch.nn.Linear(256, 128)
        # self.fc3 = torch.nn.Linear(128, args.num_edge_node * 2)
        #
        # self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        #
        # if args.use_orthogonal_init:
        #     orthogonal_init(self.conv1)
        #     orthogonal_init(self.conv2)
        #     orthogonal_init(self.conv3)
        #     orthogonal_init(self.conv4)
        #     orthogonal_init(self.fc1)
        #     orthogonal_init(self.fc2)
        #     orthogonal_init(self.fc3)

        self.conv1 = torch.nn.Conv1d(args.node_state_dim, 64, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=1)

        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, args.num_edge_node * 2)

        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            orthogonal_init(self.conv1)
            orthogonal_init(self.conv2)
            orthogonal_init(self.conv3)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            # orthogonal_init(self.fc3)


    # 输入：tensor
    # state = [batch_size, num_node, num_feature]
    def forward(self, state):
        # 应 Conv1D 的要求，
        # 将 state 的维度转换为 [batch_size, num_feature, num_node]
        state = state.transpose(1, 2).contiguous()

        # state = self.activate_func(self.conv1(state))   # [batch_size, 64, N]
        # state = self.activate_func(self.conv2(state))   # [batch_size, 128, N]
        # state = self.conv3(state)                       # [batch_size, 256, N]

        # state = self.activate_func(self.conv1(state))  # [batch_size, 64, N]
        # state = self.activate_func(self.conv2(state))  # [batch_size, 128, N]
        # state = self.activate_func(self.conv3(state))
        # state = self.conv4(state)  # [batch_size, 256, N]

        state = self.activate_func(self.conv1(state))  # [batch_size, 64, N]
        state = self.activate_func(self.conv2(state))  # [batch_size, 128, N]
        state = self.conv3(state)  # [batch_size, 256, N]


        # max pooling
        state = torch.max(state, dim=2, keepdim=True)[0]
        state = state.view(-1, 256)     # [batch_size, 256]

        # 25 节点
        # state = self.activate_func(self.fc1(state))     # [batch_size, 128]
        # state = self.activate_func(self.fc2(state))     # [batch_size, 64]
        # out = self.fc3(state)

        # 40 节点
        # state = self.activate_func(self.fc1(state))     # [batch_size, 128]
        # state = self.activate_func(self.fc2(state))
        # out = self.fc3(state)

        state = self.activate_func(self.fc1(state))     # [batch_size, 128]
        out = self.fc2(state)

        return out      # [batch_size, action_dim * 2] logits

        # # 用 mask 给非法的 logits 赋一个很小的负值，让其 softmax 概率近乎为0
        # # https://github.com/vwxyzjn/ppo-implementation-details/blob/main/ppo_multidiscrete_mask.py
        # out = torch.where(mask, out, torch.tensor(-1e9).to(device))
        # probs = F.softmax(out, dim=-1)
        #
        # return probs


class Critic(nn.Module):
    def __init__(self, args):
        super(Critic, self).__init__()

        # # ------------------- 25 节点 -----------------------------------------
        # self.conv1 = torch.nn.Conv1d(args.node_state_dim, 64, kernel_size=1)
        # self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=1)
        # self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=1)
        #
        # self.fc1 = torch.nn.Linear(256, 128)
        # self.fc2 = torch.nn.Linear(128, 64)
        # self.fc3 = torch.nn.Linear(64, 1)
        #
        # self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        #
        # if args.use_orthogonal_init:
        #     orthogonal_init(self.conv1)
        #     orthogonal_init(self.conv2)
        #     orthogonal_init(self.conv3)
        #     orthogonal_init(self.fc1)
        #     orthogonal_init(self.fc2)
        #     orthogonal_init(self.fc3)

        # # ------------------- 40 节点 -----------------------------------------
        # self.conv1 = torch.nn.Conv1d(args.node_state_dim, 64, kernel_size=1)
        # self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=1)
        # self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=1)
        # self.conv4 = torch.nn.Conv1d(256, 512, kernel_size=1)
        #
        # self.fc1 = torch.nn.Linear(512, 256)
        # self.fc2 = torch.nn.Linear(256, 128)
        # self.fc3 = torch.nn.Linear(128, 1)
        #
        # self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh
        #
        # if args.use_orthogonal_init:
        #     orthogonal_init(self.conv1)
        #     orthogonal_init(self.conv2)
        #     orthogonal_init(self.conv3)
        #     orthogonal_init(self.conv4)
        #     orthogonal_init(self.fc1)
        #     orthogonal_init(self.fc2)
        #     orthogonal_init(self.fc3)

        self.conv1 = torch.nn.Conv1d(args.node_state_dim, 64, kernel_size=1)
        self.conv2 = torch.nn.Conv1d(64, 128, kernel_size=1)
        self.conv3 = torch.nn.Conv1d(128, 256, kernel_size=1)

        self.fc1 = torch.nn.Linear(256, 128)
        self.fc2 = torch.nn.Linear(128, 1)

        self.activate_func = [nn.ReLU(), nn.Tanh()][args.use_tanh]  # Trick10: use tanh

        if args.use_orthogonal_init:
            orthogonal_init(self.conv1)
            orthogonal_init(self.conv2)
            orthogonal_init(self.conv3)
            orthogonal_init(self.fc1)
            orthogonal_init(self.fc2)
            # orthogonal_init(self.fc3)

    def forward(self, state):
        state = state.transpose(1, 2).contiguous()

        # state = self.activate_func(self.conv1(state))  # [batch_size, 64, N]
        # state = self.activate_func(self.conv2(state))  # [batch_size, 128, N]
        # state = self.conv3(state)  # [batch_size, 256, N]


        # state = self.activate_func(self.conv1(state))  # [batch_size, 64, N]
        # state = self.activate_func(self.conv2(state))  # [batch_size, 128, N]
        # state = self.activate_func(self.conv3(state))
        # state = self.conv4(state)  # [batch_size, 256, N]

        state = self.activate_func(self.conv1(state))  # [batch_size, 64, N]
        state = self.activate_func(self.conv2(state))  # [batch_size, 128, N]
        state = self.conv3(state)  # [batch_size, 256, N]



        # max pooling
        state = torch.max(state, dim=2, keepdim=True)[0]
        state = state.view(-1, 256)  # [256]

        # 25 节点
        # state = self.activate_func(self.fc1(state))
        # state = self.activate_func(self.fc2(state))
        # v_s = self.fc3(state)

        # 40 节点
        # state = self.activate_func(self.fc1(state))
        # state = self.activate_func(self.fc2(state))
        # v_s = self.fc3(state)

        state = self.activate_func(self.fc1(state))
        v_s = self.fc2(state)

        return v_s


class PPO:
    def __init__(self, args, writer):
        self.writer = writer

        self.initial_actor_learning_rate = args.lr_a
        self.initial_critic_learning_rate = args.lr_c

        self.max_episode = args.max_episode
        self.action_dim = args.action_dim

        # self.batch_size = args.batch_size
        self.mini_batch_size = args.mini_batch_size

        self.lr_a = args.lr_a  # Learning rate of actor
        self.lr_c = args.lr_c  # Learning rate of critic
        self.lr_decay_factor = args.lr_decay_factor

        self.gamma = args.gamma  # Discount factor
        self.lamda = args.lamda  # GAE parameter
        self.epsilon = args.epsilon  # PPO clip parameter
        self.K_epochs = args.K_epochs  # PPO parameter
        self.entropy_coef = args.entropy_coef  # Entropy coefficient
        self.set_adam_eps = args.set_adam_eps
        self.use_grad_clip = args.use_grad_clip
        self.use_lr_decay = args.use_lr_decay
        self.use_adv_norm = args.use_adv_norm

        self.actor = Actor(args).to(device)
        self.critic = Critic(args).to(device)

        if self.set_adam_eps:  # Trick 9: set Adam epsilon=1e-5
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a, eps=1e-5)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c, eps=1e-5)
        else:
            self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=self.lr_a)
            self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=self.lr_c)

    """
        evaluate的时候，选择最高概率的作为action
    """
    def evaluate(self, state, mask, full_mask):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(device)  # batch_size = 1
        mask = torch.unsqueeze(torch.tensor(mask, dtype=torch.bool), 0).to(device)
        full_mask = torch.unsqueeze(torch.tensor(full_mask, dtype=torch.bool), 0).to(device)


        logits = self.actor(state)      # (num_node * 2)
        logits = torch.where(mask, logits, torch.tensor(-1e9).to(device))
        split_logits = torch.split(logits, [self.action_dim, self.action_dim], dim=1)      # [[num_node], [num_node]]
        split_logits = torch.stack(split_logits)        # (2, 1, action_dim)
        split_probs = F.softmax(split_logits, dim=-1)

        # 将两部分的概率各个组合相乘，得到 [num_node * num_node] 的概率分布
        full_probs = torch.mm(split_probs[0].reshape(self.action_dim, 1), split_probs[1].reshape(1, self.action_dim))
        full_probs = full_probs.flatten()

        # 通过full_mask之后，用argmax找出概率最大的 (node_a,node_r) 组合
        # masked_full_probs = torch.where(full_mask, full_probs, torch.tensor(0.).to(device))
        # action = np.argmax(masked_full_probs.detach().cpu().numpy())
        # a0 = action // self.action_dim
        # a1 = action % self.action_dim
        # return [a0, a1]

        masked_full_probs = torch.where(full_mask, full_probs, torch.tensor(0.).to(device))
        categorical = Categorical(probs=masked_full_probs)
        action = categorical.sample()
        assert full_mask[0][action]

        action = action.item()
        a0 = action // self.action_dim
        a1 = action % self.action_dim

        return [a0, a1]


    def choose_action(self, state, mask, full_mask):
        # 如果mask中没有True，那么直接返回None，表示无法选出任何合法的action
        # if (not any(mask[:self.action_dim])) or (not any(mask[self.action_dim:])):
        if not any(full_mask):
            print("[choose_action] no legal action found.")
            return None, 0

        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float), 0).to(device)  # batch_size = 1
        mask = torch.unsqueeze(torch.tensor(mask, dtype=torch.bool), 0).to(device)
        full_mask = torch.unsqueeze(torch.tensor(full_mask, dtype=torch.bool), 0).to(device)

        with torch.no_grad():
            logits = self.actor(state)     # (1, action_dim * 2)
            logits = torch.where(mask, logits, torch.tensor(-1e9).to(device))
            split_logits = torch.split(logits, [self.action_dim, self.action_dim], dim=1)
            split_logits = torch.stack(split_logits)        # (2, 1, action_dim)
            split_probs = F.softmax(split_logits, dim=-1)   # (2, 1, action_dim)

            # 将两部分的概率各个组合相乘，得到 [num_node * num_node] 的概率分布
            full_probs = torch.mm(split_probs[0].reshape(self.action_dim, 1), split_probs[1].reshape(1, self.action_dim))
            full_probs = full_probs.flatten()

            # 通过full_mask之后，构造一个Categorical对象，然后采样动作
            masked_full_probs = torch.where(full_mask, full_probs, torch.tensor(0.).to(device))
            categorical = Categorical(probs=masked_full_probs)
            action = categorical.sample()
            assert full_mask[0][action]
            log_prob = categorical.log_prob(action)

            action = action.item()
            a0 = action // self.action_dim
            a1 = action % self.action_dim

            return [a0, a1], log_prob.item()

        #     a_probs = self.actor(state, mask)
        #     split_probs = torch.split(a_probs.squeeze(0), [self.action_dim, self.action_dim])
        #     multi_categoricals = [Categorical(probs=probs) for probs in split_probs]
        #     action = torch.stack([categorical.sample() for categorical in multi_categoricals])  # 每个动作由相应的Categorical分别采样
        #     logprob = torch.stack([categorical.log_prob(a) for a, categorical in zip(action, multi_categoricals)]).sum(0)
        #
        # return action.detach().cpu().numpy(), logprob.item()


    def update(self, replay_buffer, episode):
        # print("[Update]")

        # numpy ==> tensor
        s, a, a_logprob, r, s_, dw, done, mask, full_mask = replay_buffer.numpy_to_tensor()
        s = s.to(device)    # (batch_size, num_node, state_dim)
        a = a.to(device)
        a_logprob = a_logprob.to(device)
        r = r.to(device)
        s_ = s_.to(device)
        dw = dw.to(device)
        done = done.to(device)
        mask = mask.to(device)
        full_mask = full_mask.to(device)

        batch_size = replay_buffer.count
        print("[Update] batch_size = {}, mini_batch_size = {}".format(batch_size, self.mini_batch_size))


        """
            Calculate the advantage using GAE(General Advantage Estimation)
        """
        max_adv_each_episode = []
        adv = []
        gae = 0
        with torch.no_grad():  # adv and v_target have no gradient
            vs = self.critic(s)     # (batch_size, 1)
            vs_ = self.critic(s_)   # (batch_size, 1)
            deltas = r + self.gamma * (1.0 - dw) * vs_ - vs     # (batch_size, 1)
            for delta, d in zip(reversed(deltas.flatten().cpu().numpy()), reversed(done.flatten().cpu().numpy())):
                # 记录每个 episode 最后一个 adv
                if d and len(adv) != 0:
                    max_adv_each_episode.append(adv[0])

                gae = delta + self.gamma * self.lamda * gae * (1.0 - d)
                adv.insert(0, gae)

            max_adv_each_episode.append(adv[0])

            adv = torch.tensor(adv, dtype=torch.float).view(-1, 1).to(device)   # (batch_size, 1)
            v_target = adv + vs
            if self.use_adv_norm:  # Trick 1:advantage normalization
                adv = ((adv - adv.mean()) / (adv.std() + 1e-5))

        clipfracs = []

        # Optimize policy for K epochs:
        for epoch in range(self.K_epochs):
            # Random sampling and no repetition. 'False' indicates that training will continue even if the number of samples in the last time is less than mini_batch_size
            # for index in BatchSampler(SubsetRandomSampler(range(self.batch_size)), self.mini_batch_size, False):
            for index in BatchSampler(SubsetRandomSampler(range(batch_size)), self.mini_batch_size, False):

                logits = self.actor(s[index])       # (mini_batch_size, action_dim * 2)
                logits = torch.where(mask[index], logits, torch.tensor(-1e9).to(device))
                split_logits = torch.split(logits, [self.action_dim, self.action_dim], dim=1)
                split_logits = torch.stack(split_logits)    # tuple: 2, each = (mini_batch_size, action_dim)
                split_probs = F.softmax(split_logits, dim=-1)       # [2, mini_batch_size, action_dim]

                # (mini_batch_size, action_dim, action_dim)
                full_probs = torch.stack([torch.mm(split_probs[0, i].unsqueeze(-1), split_probs[1, i].unsqueeze(-1).T) for i in range(split_probs[0].shape[0])])
                full_probs = full_probs.reshape(full_probs.shape[0], -1)    # [mini_batch_size, action_dim ** 2]

                masked_full_probs = torch.where(full_mask[index], full_probs, torch.tensor(0.).to(device))
                categorical = Categorical(probs=masked_full_probs)
                dist_entropy = categorical.entropy()
                a_logprob_now = categorical.log_prob(a[index])


                # probs = self.actor(s[index], mask[index])
                # split_probs = torch.split(probs, [self.action_dim, self.action_dim], dim=1)
                #
                # # multi_categoricals: [Categorical(probs: torch.Size([mini_batch_size, action_dim])), Categorical(probs: torch.Size([mini_batch_size, action_dim]))]
                # # dist_entropy: [mini_batch_size]
                # multi_categoricals = [Categorical(probs=p) for p in split_probs]
                # dist_entropy = torch.stack([categorical.entropy() for categorical in multi_categoricals]).sum(0)
                # a_logprob_now = torch.stack([categorical.log_prob(action_) for action_, categorical in
                #                              zip(a[index].T, multi_categoricals)]).sum(0)


                # Finding the ratio (pi_theta / pi_theta__old)
                # a/b=exp(log(a)-log(b))
                ratios = torch.exp(a_logprob_now - a_logprob[index])  # shape(mini_batch_size X 1)

                clipfracs += [((ratios - 1.0).abs() > self.epsilon).float().mean().item()]        # 用于记录

                # Finding Surrogate Loss
                surr1 = ratios * adv[index]  # Only calculate the gradient of 'a_logprob_now' in ratios
                surr2 = torch.clamp(ratios, 1 - self.epsilon, 1 + self.epsilon) * adv[index]
                actor_loss = -torch.min(surr1, surr2) - self.entropy_coef * dist_entropy  # shape(mini_batch_size X 1)

                # Update actor
                self.optimizer_actor.zero_grad()
                actor_loss.mean().backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)
                self.optimizer_actor.step()


                entropy_loss = dist_entropy.mean()  # 用于记录

                v_s = self.critic(s[index])
                critic_loss = F.mse_loss(v_target[index], v_s)
                # Update critic
                self.optimizer_critic.zero_grad()
                critic_loss.backward()
                if self.use_grad_clip:  # Trick 7: Gradient clip
                    torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 0.5)
                self.optimizer_critic.step()

        self.writer.add_scalar("losses/policy_loss", actor_loss.mean().item(), episode)
        self.writer.add_scalar("losses/value_loss", critic_loss.item(), episode)
        self.writer.add_scalar("losses/entropy_loss", entropy_loss.item(), episode)
        self.writer.add_scalar("losses/clip_fraction", np.mean(clipfracs), episode)
        self.writer.add_scalar("gae", np.mean(max_adv_each_episode), episode)


    # 降低学习率
    def lr_decay(self, episode):
        factor = 1 - episode / self.max_episode

        self.lr_a = self.initial_actor_learning_rate * factor
        self.lr_c = self.initial_critic_learning_rate * factor
        for p in self.optimizer_actor.param_groups:
            p['lr'] = self.lr_a
        for p in self.optimizer_critic.param_groups:
            p['lr'] = self.lr_c

        print("episode: {} / {}, lr_actor = {}, lr_critic = {}".format(episode, self.max_episode, self.lr_a, self.lr_c))

        return self.lr_a




