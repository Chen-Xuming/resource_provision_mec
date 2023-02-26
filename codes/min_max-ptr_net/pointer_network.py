#!/usr/bin/env python3

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch import tanh, sigmoid


class Encoder(nn.Module):
    """
    Encoder class for Pointer-Net
    """

    def __init__(self, embedding_dim,
                 hidden_dim,
                 n_layers,
                 dropout,
                 bidir):
        """
        Initiate Encoder

        :param Tensor embedding_dim: Number of embbeding channels
        :param int hidden_dim: Number of hidden units for the LSTM
        :param int n_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(Encoder, self).__init__()
        self.hidden_dim = hidden_dim//2 if bidir else hidden_dim
        self.n_layers = n_layers*2 if bidir else n_layers
        self.bidir = bidir
        self.lstm = nn.LSTM(embedding_dim,
                            self.hidden_dim,
                            n_layers,
                            dropout=dropout,
                            bidirectional=bidir)

        # Used for propagating .cuda() command
        self.h0 = Parameter(torch.zeros(1), requires_grad=False)
        self.c0 = Parameter(torch.zeros(1), requires_grad=False)

    # embedded_inputs = (256, 5, 128)
    # hidden = 两个 (4, 256, 256) 的Tensor，h0 和 c0
    def forward(self, embedded_inputs,
                hidden):
        """
        Encoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor hidden: Initiated hidden units for the LSTMs (h, c)
        :return: LSTMs outputs and hidden units (h, c)
        """

        embedded_inputs = embedded_inputs.permute(1, 0, 2)      # 维度重排列 (256, 5, 128) ==> (5, 256, 128)

        outputs, hidden = self.lstm(embedded_inputs, hidden)    # outputs = (5, 256, 512)
                                                                # hidden(h和c) 参数发生了变化

        # outputs (5, 256, 512) ===> (256, 5, 512)
        return outputs.permute(1, 0, 2), hidden

    def init_hidden(self, embedded_inputs):
        """
        Initiate hidden units

        :param Tensor embedded_inputs: The embedded input of Pointer-NEt
        :return: Initiated hidden units for the LSTMs (h, c)
        """

        batch_size = embedded_inputs.size(0)

        # Reshaping (Expanding)
        h0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)
        c0 = self.h0.unsqueeze(0).unsqueeze(0).repeat(self.n_layers,
                                                      batch_size,
                                                      self.hidden_dim)

        return h0, c0


class Attention(nn.Module):
    """
    Attention model for Pointer-Net
    """

    def __init__(self, input_dim,
                 hidden_dim):
        """
        Initiate Attention

        :param int input_dim: Input's diamention
        :param int hidden_dim: Number of hidden units in the attention
        """

        super(Attention, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.input_linear = nn.Linear(input_dim, hidden_dim)
        self.context_linear = nn.Conv1d(input_dim, hidden_dim, 1, 1)
        self.V = Parameter(torch.FloatTensor(hidden_dim), requires_grad=True)
        self._inf = Parameter(torch.FloatTensor([float('-inf')]), requires_grad=False)      # (256, 5) 每个元素初始化为 -inf
        self.tanh = tanh
        self.softmax = nn.Softmax(dim=1)

        # Initialize vector V
        nn.init.uniform_(self.V, -1, 1)

    def forward(self, input,    # (256, 512)
                context,    # (256, 5, 512)
                mask):      # (256, 5) 元素为True/False
        """
        Attention - Forward-pass

        :param Tensor input: Hidden state h
        :param Tensor context: Attention context
        :param ByteTensor mask: Selection mask
        :return: tuple of - (Attentioned hidden state, Alphas)
        """

        # (batch, hidden_dim, seq_len) (256, 512, 5)
        inp = self.input_linear(input).unsqueeze(2).expand(-1, -1, context.size(1))

        # (batch, hidden_dim, seq_len)
        context = context.permute(0, 2, 1)  # (256, 5, 512) ==> (256, 512, 5)
        ctx = self.context_linear(context)  # (256, 512, 5)

        # (batch, 1, hidden_dim)
        V = self.V.unsqueeze(0).expand(context.size(0), -1).unsqueeze(1)    # (256, 1, 512)

        # (batch, seq_len)  (256, 5)
        att = torch.bmm(V, self.tanh(inp + ctx)).squeeze(1)

        # 将访问过的点的 alpha 变成-inf，再做softmax
        # TODO: 对于我们的问题，不需要mask了，直接softmax就好
        # if len(att[mask]) > 0:
        #     att[mask] = self.inf[mask]
        alpha = self.softmax(att)       # (256, 5) 代表选取各个点的概率值
        hidden_state = torch.bmm(ctx, alpha.unsqueeze(2)).squeeze(2)    # (256, 512)

        return hidden_state, alpha

    def init_inf(self, mask_size):
        self.inf = self._inf.unsqueeze(1).expand(*mask_size)


class Decoder(nn.Module):
    """
    Decoder model for Pointer-Net
    """

    def __init__(self, embedding_dim, hidden_dim, output_length):
        """
        Initiate Decoder

        :param int embedding_dim: Number of embeddings in Pointer-Net
        :param int hidden_dim: Number of hidden units for the decoder's RNN
        :param int output_length: 输出的服务个数
        """

        super(Decoder, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.output_length = output_length

        self.input_to_hidden = nn.Linear(embedding_dim, 4 * hidden_dim)
        self.hidden_to_hidden = nn.Linear(hidden_dim, 4 * hidden_dim)
        self.hidden_out = nn.Linear(hidden_dim * 2, hidden_dim)
        self.att = Attention(hidden_dim, hidden_dim)

        # Used for propagating .cuda() command
        self.mask = Parameter(torch.ones(1), requires_grad=False)
        self.runner = Parameter(torch.zeros(1), requires_grad=False)

    """
        embedded_inputs = (256, 5, 128)
        decoder_input = (256, 128)
        hidden = 2 * (256, 512)
        context = (256, 5, 512), encoder的输出
    """
    def forward(self, embedded_inputs,
                decoder_input,
                hidden,
                context):
        """
        Decoder - Forward-pass

        :param Tensor embedded_inputs: Embedded inputs of Pointer-Net
        :param Tensor decoder_input: First decoder's input
        :param Tensor hidden: First decoder's hidden states
        :param Tensor context: Encoder's outputs
        :return: (Output probabilities, Pointers indices), last hidden state
        """

        batch_size = embedded_inputs.size(0)        # 256
        input_length = embedded_inputs.size(1)      # 5

        # (batch, seq_len)
        mask = self.mask.repeat(input_length).unsqueeze(0).repeat(batch_size, 1)        # (256, 5) 初始化全是1
        self.att.init_inf(mask.size())

        # Generating arang(input_length), broadcasted across batch_size
        runner = self.runner.repeat(input_length)   # tensor([0., 0., 0., 0., 0.], device='cuda:0')
                                                    # 遍历之后：tensor([0., 1., 2., 3., 4.], device='cuda:0')
        for i in range(input_length):
            runner.data[i] = i
        runner = runner.unsqueeze(0).expand(batch_size, -1).long()  # (256, 5), 256个 [0,1,2,3,4]

        outputs = []
        pointers = []

        def step(x, hidden):    # x = (256, 128), hidden = 2 * (256, 512)
            """
            Recurrence step function

            :param Tensor x: Input at time t
            :param tuple(Tensor, Tensor) hidden: Hidden states at time t-1
            :return: Hidden states at time t (h, c), Attention probabilities (Alpha)
            """

            # Regular LSTM
            h, c = hidden

            gates = self.input_to_hidden(x) + self.hidden_to_hidden(h)  # (256, 2048)
            input, forget, cell, out = gates.chunk(4, 1)    # 每一个是 (256, 512)

            input = sigmoid(input)
            forget = sigmoid(forget)
            cell = tanh(cell)
            out = sigmoid(out)

            c_t = (forget * c) + (input * cell)     # (256, 512)
            h_t = out * tanh(c_t)   # (256, 512)

            # Attention section
            hidden_t, output = self.att.forward(h_t, context, torch.eq(mask, 0))    # hidden_t = (256, 512);  output = (256, 5) 是各个点的概率值
            hidden_t = tanh(self.hidden_out(torch.cat((hidden_t, h_t), 1)))     # (256, 512)

            return hidden_t, c_t, output

        # Recurrence loop
        # TODO: 对于TSP问题，输出和输入的点个数相同，但是对于我们的问题不是，这里要改一下
        for _ in range(self.output_length):
            h_t, c_t, outs = step(decoder_input, hidden)
            hidden = (h_t, c_t)

            # Masking selected inputs
            masked_outs = outs * mask

            # Get maximum probabilities and indices
            max_probs, indices = masked_outs.max(1)
            one_hot_pointers = (runner == indices.unsqueeze(1).expand(-1, outs.size()[1])).float()  # 所有访问过的点标记为1，其它未访问的点的标记为0

            # Update mask to ignore seen indices
            """
                对于我们的问题，无需mask。
                其它关于mask的地方看不懂，但是mask一直不更新，始终保持全是1就行。
            """
            # mask = mask * (1 - one_hot_pointers)    # 更新mask：访问过的点mask改成0

            # Get embedded inputs by max indices
            # embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).byte()
            embedding_mask = one_hot_pointers.unsqueeze(2).expand(-1, -1, self.embedding_dim).bool()    # (256, 5, 128) True/False
            decoder_input = embedded_inputs[embedding_mask.data].view(batch_size, self.embedding_dim)   # (256, 128)

            outputs.append(outs.unsqueeze(0))
            pointers.append(indices.unsqueeze(1))

        outputs = torch.cat(outputs).permute(1, 0, 2)   # (256, 5, 5)
        pointers = torch.cat(pointers, 1)               # (256, 5)

        return (outputs, pointers), hidden


class PointerNet(nn.Module):
    """
    Pointer-Net
    """

    def __init__(self, embedding_dim,       # 128
                 hidden_dim,                # 512
                 lstm_layers,               # 2
                 output_service_num,
                 dropout,
                 bidir=False):
        """
        Initiate Pointer-Net

        :param int embedding_dim: Number of embbeding channels
        :param int hidden_dim: Encoders hidden units
        :param int lstm_layers: Number of layers for LSTMs
        :param float dropout: Float between 0-1
        :param bool bidir: Bidirectional
        """

        super(PointerNet, self).__init__()
        self.embedding_dim = embedding_dim
        self.bidir = bidir
        self.embedding = nn.Linear(5, embedding_dim)    # 5是每个服务的属性个数
        self.encoder = Encoder(embedding_dim, hidden_dim, lstm_layers, dropout, bidir)
        self.decoder = Decoder(embedding_dim, hidden_dim, output_service_num)
        self.decoder_input0 = Parameter(torch.FloatTensor(embedding_dim), requires_grad=False)      # shape = 1 * 128

        # Initialize decoder_input0
        nn.init.uniform_(self.decoder_input0, -1, 1)    # decoder_input0 各个元素初始化为 [-1, 1] 均匀分布

    """
        inputs = (256, 5, 2)  即batch_size个TSP问题
    """
    def forward(self, inputs):
        """
        PointerNet - Forward-pass

        :param Tensor inputs: Input sequence
        :return: Pointers probabilities and indices
        """

        batch_size = inputs.size(0)         # 256
        input_length = inputs.size(1)       # 5

        decoder_input0 = self.decoder_input0.unsqueeze(0).expand(batch_size, -1)    # (256, 128)

        inputs = inputs.view(batch_size * input_length, -1)     # (1280, 2)

        # self.embedding(inputs) ==> (1280, embedding_dim=128)
        # .view(batch_size, input_length, -1) ==> (256, 5, 128)
        embedded_inputs = self.embedding(inputs).view(batch_size, input_length, -1)

        encoder_hidden0 = self.encoder.init_hidden(embedded_inputs)     # 两个(4,256,256)的Tensor，元素全都是0
        encoder_outputs, encoder_hidden = self.encoder.forward(embedded_inputs, encoder_hidden0)    # encoder_outputs = (256, 5, 512)   encoder_hidden = 2 * (4, 256, 256)
        if self.bidir:
            decoder_hidden0 = (torch.cat([_ for _ in encoder_hidden[0][-2:]], dim=-1),      # 2 * (256, 512)
                               torch.cat([_ for _ in encoder_hidden[1][-2:]], dim=-1))
        else:
            decoder_hidden0 = (encoder_hidden[0][-1],
                               encoder_hidden[1][-1])
        (outputs, pointers), decoder_hidden = self.decoder.forward(embedded_inputs,
                                                                   decoder_input0,
                                                                   decoder_hidden0,
                                                                   encoder_outputs)

        return outputs, pointers