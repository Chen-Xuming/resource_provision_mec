#!/usr/bin/env python3

"""
Pytorch implementation of Pointer Network.

http://arxiv.org/pdf/1506.03134v1.pdf.
"""

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

import numpy as np
import argparse
from tqdm import tqdm

from pointer_network import PointerNet
from data_generator import ServiceDataset

from environment import *
from numpy.random import default_rng, SeedSequence

import time

import warnings
warnings.filterwarnings("ignore", category=UserWarning)

if __name__ == '__main__':
    """ ----------------------------------------------------------------------------------------
        Hyperparameter for training
    """
    parser = argparse.ArgumentParser(description="Pointer-Net For Min-Max-Delay Resource Allocation Problem")
    # Data
    parser.add_argument('--train_size', default=256*300, type=int, help='Training data size')
    parser.add_argument('--val_size', default=10000, type=int, help='Validation data size')
    parser.add_argument('--test_size', default=10000, type=int, help='Test data size')
    parser.add_argument('--batch_size', default=256, type=int, help='Batch size')
    # Train
    parser.add_argument('--nof_epoch', default=20, type=int, help='Number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    # GPU
    parser.add_argument('--gpu', default=True, action='store_true', help='Enable gpu')
    # Network
    parser.add_argument('--embedding_size', type=int, default=128, help='Embedding size')
    parser.add_argument('--hiddens', type=int, default=256, help='Number of hidden units')
    parser.add_argument('--nof_lstms', type=int, default=2, help='Number of LSTM layers')
    parser.add_argument('--dropout', type=float, default=0., help='Dropout value')
    parser.add_argument('--bidir', default=True, action='store_true', help='Bidirectional')
    params = parser.parse_args()

    """
        Use GPU
    """
    if params.gpu and torch.cuda.is_available():
        USE_CUDA = True
        print('Using GPU, %i devices.' % torch.cuda.device_count())
    else:
        USE_CUDA = False

    """ -------------------------------------------------------------------------------------------
        Environment
    """
    num_instance = 1
    num_service = 10
    env_config = {
        "num_instance": num_instance,
        "num_service_a": num_service,
        "num_service_r": num_service,
        "budget_addition": 200,
        "num_group_per_instance": 10,
        "num_user_per_group": 10,
        "min_arrival_rate": 10,
        "max_arrival_rate": 15,
        "min_price": 1,
        "max_price": 5,
        "trigger_probability": 0.2,
        "tx_ua_min": 4,
        "tx_ua_max": 6,
        "tx_aq_min": 2,
        "tx_aq_max": 4,
        "tx_qr_min": 2,
        "tx_qr_max": 4,
        "tx_ru_min": 4,
        "tx_ru_max": 6
    }
    seed_sequence = 135792468
    rng = default_rng(seed_sequence)
    environment = Env(env_config, rng, seed_sequence)
    output_sequence_length = env_config["budget_addition"]
    print("--------------------------------------")
    print("num_user = {}".format(env_config["num_group_per_instance"] * env_config["num_user_per_group"]))
    print("budget = {}".format(env_config["budget_addition"]))
    print("num_service = {}".format(environment._num_service))
    print("seed = {}".format(seed_sequence))
    print("--------------------------------------")

    # model, dataset, loss & optimizer
    service_dataset = ServiceDataset(data_size=params.train_size, env=environment)
    dataloader = DataLoader(dataset=service_dataset, batch_size=params.batch_size, shuffle=True, num_workers=1)
    model = PointerNet(params.embedding_size, params.hiddens, params.nof_lstms, output_sequence_length, params.dropout, params.bidir)
    if USE_CUDA:
        model.cuda()
        net = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
        cudnn.benchmark = True
    CCE = torch.nn.CrossEntropyLoss()
    model_optim = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=params.lr)
    lr_decay_factor = 0.94

    # tensorboard and model saving
    date_time = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    logfile_name = date_time + "_seed_" + str(seed_sequence)
    tensorboard_logfile = "./tensorboard_writer/" + logfile_name
    model_file = "./model_checkpoint/" + logfile_name + ".pth"
    writer = SummaryWriter(tensorboard_logfile)

    """
        训练
        
        1. 初始化 best_so_far_max_delay 为无穷大，best_so_far_solution 初始化 None
        2. 抽取 batch_size 个样本输入 model ，得到 batch_size 个解
        3. 将这些解依次输入到 environment 里面，得到各个max_delay, 得到当前 batch 里的 best_solution 及其 max_delay
        4. 根据第三步的结果，更新 best_so_far_max_delay 和 best_so_far_solution（必要时）
        5. 用当前的输出和 best_so_far_solution 计算 loss
        6. 更新模型参数
    """
    step = 0
    step_losses = []
    best_so_far_max_delay = float("inf")
    best_so_far_solution = None
    for epoch in range(params.nof_epoch):
        batch_loss = []
        iterator = tqdm(dataloader, unit='Batch')

        """
            sample_batched = batch_size=128 * (service_num * 5)
        """
        for i_batch, sample_batched in enumerate(iterator):
            iterator.set_description('Epoch %i/%i' % (epoch + 1, params.nof_epoch))

            train_batch = Variable(sample_batched)

            if USE_CUDA:
                train_batch = train_batch.cuda()

            """ step 2 """
            probs, solutions = model.forward(train_batch)     # 概率分布(batch_size, 服务个数k, 服务个数)；输出的各个服务的下标（batch_size, 服务个数）
            probs = probs.contiguous().view(-1, probs.size()[-1])  # probs = (batch_size, k, k) ===> (batch_size * k, k)

            """ step 3 """
            best_so_far_max_delay_in_batch = float("inf")
            best_so_far_solution_in_batch = None
            for solution in solutions:
                max_delay, allocated_server = service_dataset.validate_solution(solution)
                if max_delay < best_so_far_max_delay_in_batch:
                    best_so_far_max_delay_in_batch = max_delay
                    best_so_far_solution_in_batch = solution

            """ step 4 """
            if best_so_far_max_delay_in_batch < best_so_far_max_delay:
                best_so_far_max_delay = best_so_far_max_delay_in_batch
                best_so_far_solution = best_so_far_solution_in_batch.repeat(sample_batched.size(0), 1).cuda()

            best_so_far_solution = best_so_far_solution.view(-1)  # (batch_size, k) ==> (batch_size * k,)


            """ step 5 """
            loss = CCE(probs, best_so_far_solution)
            step_losses.append(loss.item())
            batch_loss.append(loss.item())

            """ step 6 """
            model_optim.zero_grad()
            loss.backward()
            model_optim.step()

            iterator.set_postfix({'loss': '{%.6f}' % loss.item(), 'best_so_far_max_delay': best_so_far_max_delay})
            #iterator.set_postfix(loss='{}'.format(loss.item()))
            step_losses.append(loss.item())
            writer.add_scalar("train_loss", loss.item(), step)
            step += 1

            # model每更新100次保存1次
            if step % 100 == 0:
                torch.save(model.state_dict(), model_file)


        iterator.set_postfix({'batch_loss': '{%.6f}' % np.average(batch_loss), 'best_so_far_max_delay': best_so_far_max_delay})
        # iterator.set_postfix({'batch_loss': np.average(batch_loss), 'best_so_far_max_delay': best_so_far_max_delay})

        # 更新学习率
        model_optim.param_groups[0]['lr'] = model_optim.param_groups[0]['lr'] * lr_decay_factor
        print("lr = {}".format(model_optim.param_groups[0]['lr']))