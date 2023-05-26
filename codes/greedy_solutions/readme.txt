
1. 目的：收集 greedy 算法的解，通过 supervised 的方式训练一个模型，随后用这个预训练模型训练 RL，期望 RL 方法的性能更接近 greedy.

2. 一个文件对应于一组用户的解，命名方式为 {user_seed}_{num_user}.txt
   文件内容如下：
   第一行：user_seed = .........
   第二行：num_user = ...

   接下来的 num_user 行格式如下，其中 node_b 应当是一个固定值，否则收集程序出错。
   1 node_a node_b node_r
   2 node_a node_b node_r
   ...

   最后三行如下：
   cost = ...           # 开销
   avg_delay = ...      # 平均交互时延
   running_time = ...   # 运行时间（ms）

3. 若算法运行过程中出现无解的情况，则抛弃该样本。

4. 对于服务B位置的决策，使用与 Nearest 相同的决策

5. environment 参数
    environment_configuration = {
        "num_user": 50, (无用)
        "num_edge_node": 25,
        "minmax_arrival_rate": (5, 15),             #
        "minmax_service_rate_A": (50, 70),          # [14.3, 20] ms
        "minmax_service_rate_B": (150, 200),        # [5, 6.6] ms
        "minmax_service_rate_R": (30, 45),          # [22.2, 33.3] ms   ==> tp = [41, 60]
        "minmax_price_A": (2, 5),
        "minmax_price_B": (5, 10),
        "minmax_price_R": (4, 8),
        "price_times_of_extra_server": 2,           # 超出容量的服务器，价格是原价的若干倍
        "minmax_edge_node_capacity": (20, 40),
        "minmax_transmission_delay": (5, 16),       # tx = [20, 64] ms,  tx + tp = [61, 124]
        "minmax_transmission_price": (2, 5),
        "transmission_data_size": (1, 1, 1, 5),
        "trigger_probability": 0.6,
        "delay_limit": 0.100   # 秒
    }

    env_seed = 888888