import torch
import numpy as np
import argparse
from codes.min_cost_v4.ppo.ppo import PPO
from codes.min_cost_v4.ppo.packaged_env import PackagedEnv

from numpy.random import SeedSequence
from codes.min_cost_v4.parameters import environment_configuration as env_config
from codes.min_cost_v4.env.environment import Environment
from codes.min_cost_v4.env.user import User

import time


"""
    参数设置
"""
parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
parser.add_argument("--max_episode", type=int, default=5000, help=" Maximum number of training steps")
parser.add_argument("--save_freq", type=int, default=10, help="Save frequency")  # 每k个episodes保存一次
parser.add_argument("--update_freq", type=int, default=10, help="Update frequency")  # 每k个episodes更新一次
parser.add_argument("--lr_decay_freq", type=int, default=10, help="Learning rate decay frequency")  # 每k个episodes学习率降低
parser.add_argument("--lr_decay_factor", type=float, default=0.995, help="Learning rate decay factor")
parser.add_argument("--lr_a", type=float, default=5e-4, help="Learning rate of actor")
parser.add_argument("--lr_c", type=float, default=5e-4, help="Learning rate of critic")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
parser.add_argument("--K_epochs", type=int, default=6, help="PPO parameter")
parser.add_argument("--use_adv_norm", type=bool, default=True, help="Trick 1:advantage normalization")
parser.add_argument("--use_state_norm", type=bool, default=False, help="Trick 2:state normalization")
parser.add_argument("--use_reward_norm", type=bool, default=False, help="Trick 3:reward normalization")
parser.add_argument("--use_reward_scaling", type=bool, default=True, help="Trick 4:reward scaling")
parser.add_argument("--entropy_coef", type=float, default=0.01, help="Trick 5: policy entropy")
parser.add_argument("--use_lr_decay", type=bool, default=True, help="Trick 6:learning rate Decay")
parser.add_argument("--use_grad_clip", type=bool, default=True, help="Trick 7: Gradient clip")
parser.add_argument("--use_orthogonal_init", type=bool, default=True, help="Trick 8: orthogonal initialization")
parser.add_argument("--set_adam_eps", type=float, default=True, help="Trick 9: set Adam epsilon=1e-5")
parser.add_argument("--use_tanh", type=float, default=True, help="Trick 10: tanh activation function")

args = parser.parse_args()

args.node_state_dim = 5

args.action_dim = env_config["num_edge_node"]  # action 的取值范围是 [0, num_edge_node]
args.num_edge_node = env_config["num_edge_node"]

# args.batch_size = int(args.update_freq * env_config["num_user"])
# args.mini_batch_size = int(args.batch_size // args.save_freq)

args.batch_size = 640
args.mini_batch_size = 64


"""
    用户数量的取值范围
"""
minmax_num_user = (50, 50)

"""
    测试预训练模型
"""
def RL(env_seed_sequence, user_seed, user_num):
    print("user_seed = {}, user_num = {}".format(user_seed, user_num))

    """
        初始化环境
    """
    # env_seed_sequence = SeedSequence(env_seed)
    env = PackagedEnv(Environment(env_config, seed_sequence=env_seed_sequence))

    env.reset(user_seed=user_seed, num_user=user_num)

    """
        加载模型
    """
    checkpoint_path = "F:/resource_provision_mec/codes/min_cost_v4/ppo/checkpoint/checkpoint_20230523_212701/episode_10k.pt"
    # checkpoint_path = "F:/resource_provision_mec/codes/min_cost_v4/ppo/checkpoint/checkpoint_20230522_224534/episode_10k.pt"

    checkpoint = torch.load(checkpoint_path)
    agent = PPO(args, None)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])

    start_time = time.time()

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

    end_time = time.time()

    cost = env.env.compute_cost()
    print("final cost = {}, assigned_user: {}/{}".format(cost, len(env.assigned_users), env.env.num_user))
    return cost, (end_time - start_time) * 1000

if __name__ == '__main__':
    from numpy import random

    eseed = 888888
    useed = random.randint(0, 1000000000)
    user_num = random.randint(minmax_num_user[0], minmax_num_user[1] + 1)
    RL(eseed, useed, user_num)

