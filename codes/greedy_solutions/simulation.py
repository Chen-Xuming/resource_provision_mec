import os
import sys
sys.path.append("F:/resource_provision_mec")

from codes.greedy_solutions.parameters import environment_configuration
from numpy.random import SeedSequence
from codes.greedy_solutions.algorithms.greedy import GreedyAssignmentAllocation
from codes.greedy_solutions.env.environment import Environment
from numpy import random



"""
    将一组解写入单个文件
"""
def write_result_to_file(user_seed, num_user, solution, cost, avg_delay, running_time):
    save_dir = "./solutions/user_{}".format(num_user)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    save_path = save_dir + "/" + "{}_{}.txt".format(user_seed, num_user)

    with open(save_path, 'w') as file:
        file.write("user_seed = {}\n".format(user_seed))
        file.write("num_user = {}\n".format(num_user))

        for i, line in enumerate(solution):
            file.write("{} {} {} {}\n".format(i, line[0], line[1], line[2]))

        file.write("cost = {}\n".format(cost))
        file.write("avg_delay = {}\n".format(avg_delay))
        file.write("running_time = {}\n".format(running_time))



"""
   各用户数轮流跑，不断循环 
"""
simulation_no = 0

env_seed = 888888
user_range = (40, 50)
user_step = 1

simulation_times = 20000

print("=============================================")
print("simulation_no = ", simulation_no)
print("env_seed = ", env_seed)
print("user_range = ", user_range)
print("num_edge_node = ", environment_configuration["num_edge_node"])
print("simulation_times = ", simulation_times)
print("=============================================")

if __name__ == '__main__':
    for i in range(simulation_times):
        for u_num in range(user_range[0], user_range[1] + user_step, user_step):
            user_seed = random.randint(0, 1000000000)
            print("\n[sim_no = {}, u_num = {}, u_seed = {}]".format(i, u_num, user_seed))

            env_seed_sequence = SeedSequence(env_seed)
            env = Environment(environment_configuration, env_seed_sequence)
            env.set_users_and_services_by_given_seed(user_seed=user_seed, num_user=u_num)

            greedy_alg = GreedyAssignmentAllocation(env)
            success = greedy_alg.run()
            if not success:
                continue

            solution, cost, avg_delay, running_time = greedy_alg.get_solution()
            write_result_to_file(user_seed=user_seed,
                                 num_user=u_num,
                                 solution=solution,
                                 cost=cost,
                                 avg_delay=avg_delay,
                                 running_time=running_time)



















