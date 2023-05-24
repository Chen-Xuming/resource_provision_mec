import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from codes.min_cost_unsharedA.ppo.normalization import Normalization, RewardScaling
from codes.min_cost_unsharedA.ppo.replay_buffer import ReplayBuffer
from codes.min_cost_unsharedA.ppo.ppo import PPO
from codes.min_cost_unsharedA.ppo.packaged_env import PackagedEnv

from numpy.random import SeedSequence
from codes.min_cost_unsharedA.parameters import environment_configuration as env_config
from codes.min_cost_unsharedA.env.environment import Environment

parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
parser.add_argument("--max_train_steps", type=int, default=int(2e6), help=" Maximum number of training steps")
parser.add_argument("--evaluate_freq", type=float, default=2e3, help="Evaluate the policy every 'evaluate_freq' steps")
parser.add_argument("--save_freq", type=int, default=2e4, help="Save frequency")
parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
parser.add_argument("--hidden_width", type=int, default=64,
                    help="The number of neurons in hidden layers of the neural network")
parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
parser.add_argument("--K_epochs", type=int, default=10, help="PPO parameter")
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

parser.add_argument("--node_embedding_size", type=int, default=30, help="Model parameter")
parser.add_argument("--user_embedding_size", type=int, default=200, help="Model parameter")
parser.add_argument("--hidden_dim", type=int, default=1024, help="Model parameter")

args = parser.parse_args()

env_config["num_user"] = 75

def RL(env_seed, user_seed):
    original_env = Environment(env_config, SeedSequence(env_seed))
    original_env.reset_parameters_about_users(user_seed)

    packaged_env = PackagedEnv(original_env)
    np.random.seed(env_seed)
    torch.manual_seed(env_seed)

    args.node_state_dim = (env_config["num_edge_node"], packaged_env.node_state_dim)
    args.user_state_dim = packaged_env.user_state_dim
    args.action_dim = env_config["num_edge_node"]
    args.max_episode_steps = env_config["num_edge_node"]

    checkpoint_path = "F:/resource_provision_mec/codes/min_cost_unsharedA/ppo/checkpoint/20230420_230614_seed_1136447707.pt"
    checkpoint = torch.load(checkpoint_path)
    agent = PPO(args, packaged_env)
    agent.actor.load_state_dict(checkpoint["actor_state_dict"])
    agent.critic.load_state_dict(checkpoint["critic_state_dict"])

    done = False
    graph_s, user_s = packaged_env.reset()
    while not done:
        action = agent.evaluate(graph_s, user_s)
        graph_s_, user_s_, r, done, _ = packaged_env.step(action)
        cur_user = packaged_env.assigned_users[-1]
        print("[assign] user {} to ({}, {}, {})".format(cur_user.user_id,
                                                        cur_user.service_A.node_id,
                                                        cur_user.service_B.node_id,
                                                        cur_user.service_R.node_id))

        graph_s = graph_s_
        user_s = user_s_

    cost = packaged_env.env.compute_cost(packaged_env.assigned_users)
    print("Final Cost =  {}, assigned_user: {} / {}".format(cost,
                                                            len(packaged_env.assigned_users),
                                                            packaged_env.env.num_user))
    return cost


if __name__ == '__main__':
    env_seed = 1136447707
    RL(env_seed, user_seed=1)

    # for useed in range(1):
    #     print("simulation: useed = ", useed)
    #     RL(env_seed, useed)