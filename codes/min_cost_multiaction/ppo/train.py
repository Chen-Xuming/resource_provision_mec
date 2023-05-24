import torch
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import gym
import argparse
from codes.min_cost_multiaction.ppo.normalization import Normalization, RewardScaling
from codes.min_cost_multiaction.ppo.replay_buffer import ReplayBuffer
from codes.min_cost_multiaction.ppo.ppo import PPO
from codes.min_cost_multiaction.ppo.packaged_env import PackagedEnv

from numpy.random import SeedSequence
from codes.min_cost_multiaction.parameters import environment_configuration as env_config
from codes.min_cost_multiaction.env.environment import Environment

import time

"""
    evaluate 若干次，reward取平均值。
    注意：evaluate的时候，是有可能因为违反时延约束而提前结束的。
"""
def evaluate_policy(args, env, agent, node_state_norm, user_state_norm):
    times = 1
    evaluate_reward = 0

    info = []

    for _ in range(times):
        graph_s, user_s = env.reset()
        if args.use_state_norm:  # During the evaluating,update=False       两种state分别归一化
            # graph_s.x = node_state_norm(graph_s.x, update=False)
            user_s = user_state_norm(user_s, update=False)

        done = False
        episode_reward = 0
        while not done:
            a = agent.evaluate(graph_s, user_s)  # We use the deterministic policy during the evaluating
            graph_s_, user_s_, r, done, _ = env.step(a)
            if args.use_state_norm:
                # graph_s_.x = node_state_norm(graph_s_.x, update=False)
                user_s_ = user_state_norm(user_s_, update=False)
            episode_reward += r
            graph_s = graph_s_
            user_s = user_s_
        evaluate_reward += episode_reward

        info.append({"cost": env.env.compute_cost(env.assigned_users),
                     "assigned_user": "{} / {}".format(len(env.assigned_users), env.env.num_user)})

    return evaluate_reward / times, info

def main(args, seed):
    env = PackagedEnv(Environment(env_config, SeedSequence(seed)))
    env_evaluate = PackagedEnv(Environment(env_config, SeedSequence(seed)))  # When evaluating the policy, we need to rebuild an environment

    np.random.seed(seed)
    torch.manual_seed(seed)

    args.node_state_dim = (env_config["num_edge_node"], env.node_state_dim)
    args.user_state_dim = env.user_state_dim
    args.action_dim = env_config["num_edge_node"]
    args.max_episode_steps = env_config["num_edge_node"]  # Maximum number of steps per episode

    evaluate_num = 0  # Record the number of evaluations
    evaluate_rewards = []  # Record the rewards during the evaluating
    total_steps = 0  # Record the total steps during the training

    replay_buffer = ReplayBuffer(args)
    agent = PPO(args, env)

    date_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    writer = SummaryWriter(log_dir="data/{}_seed_{}".format(date_time, seed))

    node_state_norm = Normalization(shape=args.node_state_dim)
    user_state_norm = Normalization(shape=args.user_state_dim)
    if args.use_reward_norm:  # Trick 3:reward normalization
        reward_norm = Normalization(shape=1)
    elif args.use_reward_scaling:  # Trick 4:reward scaling
        reward_scaling = RewardScaling(shape=1, gamma=args.gamma)

    while total_steps < args.max_train_steps:
        graph_s, user_s = env.reset()
        if args.use_state_norm:
            # graph_s.x = node_state_norm(graph_s.x)
            user_s = user_state_norm(user_s)
        if args.use_reward_scaling:
            reward_scaling.reset()

        episode_steps = 0
        done = False
        while not done:
            episode_steps += 1
            a, a_logprob = agent.choose_action(graph_s, user_s)  # Action and the corresponding log probability   a_logprob是一个标量
            graph_s_, user_s_, r, done, _ = env.step(a)

            if args.use_state_norm:
                # graph_s_.x = node_state_norm(graph_s_.x)
                user_s_ = user_state_norm(user_s_)
            if args.use_reward_norm:
                r = reward_norm(r)
            elif args.use_reward_scaling:
                r = reward_scaling(r)

            # dw = True 表示违反时延约束而提前结束
            if done and episode_steps != args.max_episode_steps:
                dw = True
            else:
                dw = False

            replay_buffer.store(graph_s, graph_s_, user_s, a, a_logprob, r, dw, done)
            graph_s = graph_s_
            user_s = user_s_

            total_steps += 1

            # When the number of transitions in buffer reaches batch_size,then update
            if replay_buffer.count == args.batch_size:
                agent.update(replay_buffer, total_steps)
                replay_buffer.clear()

            if total_steps % args.evaluate_freq == 0:
                evaluate_num += 1
                evaluate_reward, info = evaluate_policy(args, env_evaluate, agent, node_state_norm, user_state_norm)
                evaluate_rewards.append(evaluate_reward)
                print("evaluate_num:{} \t evaluate_reward:{} \t info: {}".format(evaluate_num, evaluate_reward, info))
                writer.add_scalar('step_reward', evaluate_rewards[-1], global_step=total_steps)
                # Save the rewards
                if evaluate_num % args.save_freq == 0:
                    np.save('./data/{}_seed_{}.npy'.format(date_time, seed),
                            np.array(evaluate_rewards))

                torch.save({"actor_state_dict": agent.actor.state_dict(),
                            "critic_state_dict": agent.critic.state_dict()}, "checkpoint/{}_seed_{}.pt".format(date_time, seed))


if __name__ == '__main__':
    parser = argparse.ArgumentParser("Hyperparameter Setting for PPO-discrete")
    parser.add_argument("--max_train_steps", type=int, default=int(4e6), help=" Maximum number of training steps")
    parser.add_argument("--evaluate_freq", type=float, default=4e3, help="Evaluate the policy every 'evaluate_freq' steps")
    parser.add_argument("--save_freq", type=int, default=2e4, help="Save frequency")
    parser.add_argument("--batch_size", type=int, default=1024, help="Batch size")
    parser.add_argument("--mini_batch_size", type=int, default=64, help="Minibatch size")
    parser.add_argument("--hidden_width", type=int, default=64, help="The number of neurons in hidden layers of the neural network")
    parser.add_argument("--lr_a", type=float, default=3e-4, help="Learning rate of actor")
    parser.add_argument("--lr_c", type=float, default=3e-4, help="Learning rate of critic")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--lamda", type=float, default=0.95, help="GAE parameter")
    parser.add_argument("--epsilon", type=float, default=0.2, help="PPO clip parameter")
    parser.add_argument("--K_epochs", type=int, default=5, help="PPO parameter")
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

    main(args, seed=1136447707)