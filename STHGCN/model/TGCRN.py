import torch
import torch.nn as nn
from model.TGCRNCell import TGCRNCell
from model.TimeGraph import TimeGraphRegulation
import torch.nn.functional as F
import numpy as np


class Encoder(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, num_layers, gcn_layers, time_layers, alpha,
                 droprate, time_drop):
        super(Encoder, self).__init__()
        assert num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.node_num = node_num
        self.input_dim = dim_in
        self.num_layers = num_layers
        self.dcrnn_cells = nn.ModuleList()
        self.gcn_layers = gcn_layers
        self.time_layers = time_layers
        self.alpha = alpha
        self.droprate = droprate
        self.time_drop = time_drop
        self.dcrnn_cells.append(
            TGCRNCell(node_num, dim_in, dim_out, cheb_k, embed_dim, self.gcn_layers, self.alpha, self.droprate))
        for _ in range(1, num_layers):
            self.dcrnn_cells.append(
                TGCRNCell(node_num, dim_out, dim_out, cheb_k, embed_dim, self.gcn_layers, self.alpha, self.droprate))
        self.TimeGraph = TimeGraphRegulation(64, 64, self.time_layers, self.time_drop)
        self.vff = nn.Linear(1, 64)

    def forward(self, x, init_state, node_embeddings, time_embeddings,nodevec1, nodevec2):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)

        assert x.shape[2] == self.node_num and x.shape[3] == self.input_dim

        output_hidden = []
        current_inputs = self.vff(x)
        s =0
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []

            state,s= self.dcrnn_cells[i](current_inputs, state, node_embeddings,nodevec1, nodevec2)

            current_inputs = state
        current_inputs = self.TimeGraph(current_inputs, time_embeddings)

        return current_inputs, output_hidden, s

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)  # (num_layers, B, N, hidden_dim)


class TGCRN(nn.Module):
    def __init__(self, args):
        super(TGCRN, self).__init__()
        self.num_node = args.num_nodes
        self.input_dim = args.input_dim
        self.hidden_dim = args.rnn_units
        self.output_dim = args.output_dim
        self.horizon = args.horizon
        self.num_layers = args.num_layers

        self.gcn_layer = args.gcn_layer
        self.time_layer = args.time_layer
        self.droprate = args.droprate
        self.time_drop = args.time_drop
        self.alpha = args.alpha
        self.time_dim = args.time_dim
        self.default_graph = args.default_graph
        self.node_embeddings = nn.Parameter(torch.randn(self.num_node, args.embed_dim), requires_grad=True)
        self.time_embeddings = nn.Parameter(torch.randn(self.horizon, self.time_dim), requires_grad=True)
        self.nodevec1 = nn.Parameter(torch.randn(self.num_node, args.embed_dim),
                                     requires_grad=True)
        self.nodevec2 = nn.Parameter(torch.randn(args.embed_dim, self.num_node),
                                     requires_grad=True)
        self.encoder = Encoder(args.num_nodes, args.input_dim, args.rnn_units, args.cheb_k,
                               args.embed_dim, args.num_layers, self.gcn_layer, self.time_layer, self.alpha,
                               self.droprate, self.time_drop)

        # predictor
        self.end_conv = nn.Conv2d(1, args.horizon * self.output_dim * 3, kernel_size=(1, self.hidden_dim), bias=True)
        self.end_conv_2 = nn.Conv2d(in_channels=args.horizon * self.output_dim * 3,
                                    out_channels=args.horizon * self.output_dim,
                                    kernel_size=(1, 1),
                                    bias=True)
    def forward(self, source, targets, teacher_forcing_ratio=0.5):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _, s = self.encoder(source, init_state, self.node_embeddings, self.time_embeddings, self.nodevec1, self.nodevec2)  # B, T, N, hidden
        output = output[:, -1:, :, :]  # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv((output))  # B, T*C, N, 1
        output = self.end_conv_2((output))
        output = output.squeeze(-1).reshape(-1, self.horizon, self.output_dim, self.num_node)
        output = output.permute(0, 1, 3, 2)  # B, T, N, C

        return output,s