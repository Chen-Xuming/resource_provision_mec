import os
import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import argparse
from codes.min_cost_v4.ppo.replay_buffer import ReplayBuffer
from codes.min_cost_v4.ppo.ppo import PPO
from codes.min_cost_v4.ppo.packaged_env import PackagedEnv

from numpy.random import SeedSequence
from codes.min_cost_v4.parameters import environment_configuration as env_config
from codes.min_cost_v4.env.environment import Environment

from codes.min_cost_v4.algorithms.nearest import NearestAssignmentAllocation

from codes.min_cost_v4.ppo.normalization import RewardScaling

import time
from numpy import random

"""
    用户数量的取值范围
"""
minmax_num_user = (40, 50)

"""
    验证当前模型的性能
"""
def evaluate_policy(env_seed, agent):
    user_seed = random.randint(0, 10000000)
    user_num = random.randint(minmax_num_user[0], minmax_num_user[1] + 1)

    print("\n============== Evaluation: user_seed: {}, num_user: {} ================".format(user_seed, user_num))
    env_seed_sequence = SeedSequence(env_seed)
    env = PackagedEnv(Environment(env_config, seed_sequence=env_seed_sequence))
    env.reset(user_seed=user_seed, num_user=user_num)

    done = False
    mask, full_mask = env.get_mask()
    state = env.get_state(done=False, mask=mask)
    user_id = 0
    while not done:
        action = agent.evaluate(state, mask, full_mask)

        assert action is not None, "action is None!"

        state_, mask_, full_mask_, reward, done, _ = env.step(action)
        user = env.env.user_list[user_id]   # type: User
        print("[assign user {}] --> ({}, {}, {})".format(user_id, user.service_A.node_id, user.service_B.node_id, user.service_R.node_id))

        state = state_
        mask = mask_
        full_mask = full_mask_

        user_id += 1

    cost_rl = env.env.compute_cost()

    nearest_env = Environment(env_config, env_seed_sequence)
    nearest_env.set_users_and_services_by_given_seed(user_seed=user_seed, num_user=user_num)
    nearest_alg = NearestAssignmentAllocation(nearest_env)
    nearest_alg.run()
    cost_nearest = nearest_env.compute_cost()

    print("cost(RL): {}, cost(Ne): {}, ratio: {}".format(cost_rl, cost_nearest, cost_rl/cost_nearest))
    print("\n\n")

def main(args, env_seed):
    date_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    writer = SummaryWriter(log_dir="data/{}_env-seed_{}".format(date_time, env_seed))

    print("==================")
    print(date_time)
    print("==================")

    replay_buffer = ReplayBuffer(args)
    agent = PPO(args, writer)

    # 底层网络配置固定不变，每个episode变化的是用户的信息
    env_seed_sequence = SeedSequence(env_seed)
    env = PackagedEnv(Environment(env_config, seed_sequence=env_seed_sequence))

    if args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)


    episode = 1
    while episode <= args.max_episode:
        user_seed = random.randint(0, 10000000)
        user_num = random.randint(minmax_num_user[0], minmax_num_user[1] + 1)
        env.reset(user_seed=user_seed, num_user=user_num)

        mask, full_mask = env.get_mask()
        state = env.get_state(done=False, mask=mask)

        if args.use_reward_scaling:
            reward_scaling.reset()

        episode_reward = 0
        done = False
        while not done:
            action, a_logprob = agent.choose_action(state, mask, full_mask)  # Action and the corresponding log probability   a_logprob是一个标量
            state_, mask_, full_mask_, reward, done, _ = env.step(action)      # 此处的state_已经归一化

            if args.use_reward_scaling:
                reward = reward_scaling(reward)

            # dw = True 表示无法选出合法操作而提前结束
            if done and len(env.assigned_users) < env.env.num_user:
                dw = True
            else:
                dw = False

            if action is not None:
                stored_a = action[0] * args.num_edge_node + action[1]   #
                replay_buffer.store(state, stored_a, a_logprob, reward, state_, dw, done, mask, full_mask)
                state = state_
                mask = mask_
                full_mask = full_mask_

            episode_reward += reward

        # 记录episode的reward
        writer.add_scalar("episode_reward", episode_reward, global_step=episode)

        # 记录和nearest算法cost的比值
        cost_rl = env.env.compute_cost()
        nearest_env = Environment(env_config, env_seed_sequence)
        nearest_env.set_users_and_services_by_given_seed(user_seed=user_seed, num_user=user_num)
        nearest_alg = NearestAssignmentAllocation(nearest_env)
        nearest_alg.run()
        cost_nearest = nearest_env.compute_cost()
        cost_ratio = cost_rl / cost_nearest
        writer.add_scalar("episode_cost_ratio", cost_ratio, global_step=episode)

        print("[episode {}] reward = {}, ratio = {:.3f} ({} / {}), num_user = {}, sum_lambda = {}, user_seed = {}".format(episode, episode_reward, cost_rl / cost_nearest, cost_rl, cost_nearest, user_num, env.env.total_arrival_rate, user_seed))
        # print("[episode {}] reward = {}".format(episode, episode_reward))


        # 定期更新模型更新模型
        if episode % args.update_freq == 0:
            agent.update(replay_buffer, episode)
            replay_buffer.clear_all()

        # 定期保存模型
        if episode % args.save_freq == 0:
            checkpoint_dir = "./checkpoint/checkpoint_{}".format(date_time)
            if not os.path.exists(checkpoint_dir):
                os.mkdir(checkpoint_dir)
            episode_k = episode // 1000
            torch.save({"actor_state_dict": agent.actor.state_dict(),
                        "critic_state_dict": agent.critic.state_dict()},
                       "./checkpoint/checkpoint_{}/episode_{}k.pt".format(date_time, episode_k))

        # 定期更新学习率
        if args.use_lr_decay and episode % args.lr_decay_freq == 0:
            cur_lr = agent.lr_decay(episode)
            writer.add_scalar("learning_rate", cur_lr, global_step=episode)

        # 定期evaluate
        if episode % 20 == 0:
            evaluate_policy(env_seed, agent)

        episode += 1


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_episode", type=int, default=10000, help=" Maximum number of training steps")
    parser.add_argument("--save_freq", type=int, default=200, help="Save frequency")      # 每k个episodes保存一次
    parser.add_argument("--update_freq", type=int, default=15, help="Update frequency")  # 每k个episodes更新一次
    parser.add_argument("--lr_decay_freq", type=int, default=15, help="Learning rate decay frequency")      # 每k个episodes学习率降低
    parser.add_argument("--lr_decay_factor", type=float, default=0.995, help="Learning rate decay factor")
    parser.add_argument("--lr_a", type=float, default=4e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=4e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=6, help="PPO parameter")
    parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")      # original = True
    parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
    parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
    parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")        # original = True
    parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
    parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
    parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
    parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
    parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
    parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

    args = parser.parse_args()

    args.node_state_dim = 5

    args.action_dim = env_config["num_edge_node"]       # action 的取值范围是 [0, num_edge_node]
    args.num_edge_node = env_config["num_edge_node"]

    args.batch_size = int(args.update_freq * env_config["num_user"])
    # args.batch_size = 640
    args.mini_batch_size = 50

    env_seed = 888888
    main(args, env_seed)