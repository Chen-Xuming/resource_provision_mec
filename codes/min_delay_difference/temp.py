from algorithm import Algorithm
from time import time

class greedy_temp(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)
        self.iteration = 1

    def set_num_server(self):
        self._solutions = []

        self._start_time = time()
        self.get_initial_max_interaction_delay()

        while self._cost < self._env._cost_budget:
            max_delay, user_i, user_j = self._env.get_max_interaction_delay()

            user_i = self._env._users[user_i]
            user_j = self._env._users[user_j]
            service_type = ["a", "q", "r"]
            service_list = [
                self._env._service_A[user_i._service_a],
                self._env._service_q,
                self._env._service_R[user_j._service_r][user_j._sub_service_r]
            ]

            services = []
            for i, s_type in enumerate(service_type):
                # 这里计算的增加一台服务器减少的时延量，与优化目标考虑的是整个时延还是排队时延没有关系。
                services.append({"service": s_type,
                                 "id": (service_list[i]._id, service_list[i]._sub_id),
                                 "reduction": service_list[i].reduction_of_delay_when_add_a_server(),
                                 "price": service_list[i]._price
                                 })

            max_utility = self.get_max_utility(services)

            if max_utility is not None:
                selected_service = service_list[service_type.index(max_utility)]
                selected_service.update_num_server(selected_service._num_server + 1)

                a, q, r, sub_r = self._env.get_service_index(user_i._id, user_j._id)
                indices = (a, q, (r, sub_r))
                self._solutions.append([max_utility, indices[service_type.index(max_utility)], max_delay, (user_i._id, user_j._id)])
            else:
                break
            self._cost = self._env.compute_cost()

            max_delay, user_i, user_j = self._env.get_max_interaction_delay()
            # print("max_delay = {}".format(max_delay))

            self.iteration += 1

        self.get_min_difference_result()
        self.get_running_time()

    def get_max_utility(self, services):
        services = sorted(services, key=lambda services: services["reduction"] / services["price"], reverse=True)
        max_utility = None

        for k in range(len(services)):
            if services[k]["price"] > self._env._cost_budget - self._cost:
                continue
            else:
                max_utility = services[k]["service"]
                break
        return max_utility

"""
    针对 min (T_max - T_min) 的算法
"""
class delay_balance(Algorithm):
    def __init__(self, env):
        Algorithm.__init__(self, env)
        self._threshold = 0.120   #时延阈值是150ms
        self._cost = self._env.compute_cost()

    def DEBUG(self, info):
        on_off = True
        if on_off:
            print(info)

    def set_num_server(self):
        self._start_time = time()

        """
            step 1: 将所有时延降到阈值以下
        """
        finish_flag = False
        max_delay, min_delay, count_large_delay = self.get_max_delay_and_min_delay()
        self.DEBUG("count_large_delay = {} max = {}, min = {}, difference = {}".format(count_large_delay,
                                                                                       max_delay[1],
                                                                                       min_delay[1],
                                                                                       max_delay[1] - min_delay[1]))
        while count_large_delay > 0 and not finish_flag:
            self.DEBUG("count_large_delay = {} max = {}, min = {}, difference = {}".format(count_large_delay,
                                                                                           max_delay[1],
                                                                                           min_delay[1],
                                                                                           max_delay[1] - min_delay[1]))

            max_service_chain = self._env.get_service_index(max_delay[0][0], max_delay[0][1])
            min_service_chain = self._env.get_service_index(min_delay[0][0], min_delay[0][1])

            max_delay_reductions = self.compute_delay_reductions(max_service_chain)
            min_delay_reductions = self.compute_delay_reductions(min_service_chain)

            # 选择max_service_chain中的某个服务，为其增加一个服务器
            # 如果T_min > threshold, 那么选择减小量最大的
            # 如果T_min <= threshold, 设 s = [s1, ..., sk] 是一组能让 T_max 降到阈值以下的服务，
            # 从中选择对 T_min 影响最小的那个；若s为空，选择减少量最大的
            if min_delay[1] > self._threshold:
                finish_flag = self.allocate_for_max_reduction_service(max_delay_reductions)

            else:
                candidates = []
                for service in max_delay_reductions:
                    if max_delay[1] - service[1] <= self._threshold:
                        candidates.append(service)
                if len(candidates) == 0:
                    finish_flag = self.allocate_for_max_reduction_service(max_delay_reductions)
                else:
                    finish_flag = self.allocate_for_min_influence_and_max_reduction(min_delay_reductions, candidates)

            self._cost = self._env.compute_cost()
            max_delay, min_delay, count_large_delay = self.get_max_delay_and_min_delay()

        """
            step 2: 对最大时延做对最小时延影响最小的调整
        """
        self.DEBUG("--------------------- step 2 ---------------------------")
        while self._cost < self._env._cost_budget and not finish_flag:
            max_delay, min_delay, count_large_delay = self.get_max_delay_and_min_delay()
            self.DEBUG("max = {}, min = {}, difference = {}".format(max_delay[1], min_delay[1], max_delay[1] - min_delay[1]))

            if max_delay[1] - min_delay[1] < 1e-5:
                break
            max_service_chain = self._env.get_service_index(max_delay[0][0], max_delay[0][1])
            min_service_chain = self._env.get_service_index(min_delay[0][0], min_delay[0][1])
            max_delay_reductions = self.compute_delay_reductions(max_service_chain)
            min_delay_reductions = self.compute_delay_reductions(min_service_chain)

            finish_flag = self.allocate_for_min_influence_and_max_reduction(min_delay_reductions, max_delay_reductions)
            self._cost = self._env.compute_cost()

        """
            资源归还
        """
        self.DEBUG("--------------------- step 3 ---------------------------")
        finish_flag = False
        while not finish_flag:
            max_delay, min_delay, count_large_delay = self.get_max_delay_and_min_delay()
            self.DEBUG("max user pair: ({}, {})".format(max_delay[0][0], max_delay[0][1]))
            self.DEBUG(
                "max = {}, min = {}, difference = {}".format(max_delay[1], min_delay[1], max_delay[1] - min_delay[1]))
            max_service_chain = self._env.get_service_index(max_delay[0][0], max_delay[0][1])
            min_service_chain = self._env.get_service_index(min_delay[0][0], min_delay[0][1])
            self.DEBUG("max_chain: {}, min_chain: {}".format(max_service_chain, min_service_chain))
            selected_service = None
            max_increment = 0
            if min_service_chain[0] != max_service_chain[0]:
                increment_a = self._env._service_A[min_service_chain[0]].delay_increment_when_reduce_a_server()
                self.DEBUG("increment_a = {}, min_delay = {}, sum = {}, max_delay = {}".format(increment_a, min_delay[1], increment_a + min_delay[1], max_delay[1]))
                if increment_a != 0 and increment_a > max_increment and increment_a + min_delay[1] <= max_delay[1]:
                    selected_service = self._env._service_A[min_service_chain[0]]
                    max_increment = increment_a
            if not(min_service_chain[2] == max_service_chain[2] and min_service_chain[3] == max_service_chain[3]):
                increment_r = self._env._service_R[min_service_chain[2]][min_service_chain[3]].delay_increment_when_reduce_a_server()
                self.DEBUG(
                    "increment_r = {}, min_delay = {}, sum = {}, max_delay = {}".format(increment_r, min_delay[1],
                                                                                        increment_r + min_delay[1],
                                                                                        max_delay[1]))
                if increment_r != 0 and increment_r > max_increment and increment_r + min_delay[1] <= max_delay[1]:
                    selected_service = self._env._service_R[min_service_chain[2]][min_service_chain[3]]
                    max_increment = increment_r
            if selected_service is None:
                finish_flag = True
            else:
                self.DEBUG("selected service: ({}, {}), max_increment = {}".format(selected_service._id,
                                                                                   selected_service._sub_id,
                                                                                   max_increment))
                selected_service.update_num_server(selected_service._num_server - 1)
                self._cost = self._env.compute_cost()

        """
            结束
        """
        self.get_min_difference_result()
        self.get_running_time()

    # 获取最大时延、最小时延、大于阈值的时延个数
    def get_max_delay_and_min_delay(self):
        # [((user_i, user_j), delay), ...]，倒序
        sorted_delay_list = self._env.get_sorted_interaction_delay_list()
        delays_greater_than_threshold = []
        for item in sorted_delay_list:
            if item[1] > self._threshold:
                delays_greater_than_threshold.append(item)
            else:
                break
        return sorted_delay_list[0], sorted_delay_list[-1], len(delays_greater_than_threshold)

    # 对一个服务链上的节点计算增加一个服务器减少的时延量
    # 返回排倒序的列表 [(service, reduction), ...]
    def compute_delay_reductions(self, service_chain):
        reduction_a = self._env._service_A[service_chain[0]].reduction_of_delay_when_add_a_server()
        reduction_q = self._env._service_q.reduction_of_delay_when_add_a_server()
        reduction_r = self._env._service_R[service_chain[2]][service_chain[3]].reduction_of_delay_when_add_a_server()
        res = {
            self._env._service_A[service_chain[0]] : reduction_a,
            self._env._service_q : reduction_q,
            self._env._service_R[service_chain[2]][service_chain[3]] : reduction_r
        }
        return sorted(res.items(), key=lambda x: x[1], reverse=True)

    # 返回 finish_flag
    def allocate_for_max_reduction_service(self, services):
        selected_service = None
        for service in services:
            if self._cost + service[0]._price <= self._env._cost_budget:
                selected_service = service
                break
        if selected_service is None:
            return True
        selected_service[0].update_num_server(selected_service[0]._num_server + 1)
        return False

    # 返回 finish_flag
    def allocate_for_min_influence_and_max_reduction(self, min_delay_services, candidates):
        reduction = 0.
        selected_candidate = None
        for candidate in candidates:
            if self._cost + candidate[0]._price > self._env._cost_budget:
                continue
            for min_service in min_delay_services:
                if candidate[0]._type == min_service[0]._type:
                    if (candidate[0] is not min_service[0]) and candidate[1] > reduction:
                        selected_candidate = candidate[0]
                        reduction = candidate[1]
        if selected_candidate is None:
            return True
        selected_candidate.update_num_server(selected_candidate._num_server + 1)
        self.DEBUG("[allocate_for_min_influence_and_max_reduction]: {}, {}, {}".format(selected_candidate._type, (selected_candidate._id, selected_candidate._sub_id), reduction))
        return False