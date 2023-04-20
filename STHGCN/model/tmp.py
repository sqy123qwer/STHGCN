import torch
import torch.nn.functional as F
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss

import torch
import torch.nn.functional as F
import torch.nn as nn
from logging import getLogger
from libcity.model.abstract_traffic_state_model import AbstractTrafficStateModel
from libcity.model import loss


class GCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, embed_dim, gcn_layers, alpha, droprate):
        super(GCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights_pool = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_in, dim_out))
        self.weights_pool2 = nn.Parameter(torch.FloatTensor(embed_dim, cheb_k, dim_out, dim_out))
        self.bias_pool = nn.Parameter(torch.FloatTensor(embed_dim, dim_out))
        self.gcn_layer = gcn_layers
        self.alpha = alpha
        self.droprate = droprate
    def forward(self, x, node_embeddings):
        # x shaped[B, N, C], node_embeddings shaped [N, D] -> supports shaped [N, N]
        # output shape [B, N, C]
        node_num = node_embeddings.shape[0]
        supports = F.softmax(F.relu(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))), dim=1)
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)
        weights = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)                       # N, dim_out
        x_g = torch.einsum("knm,bmc->bknc", supports, x)      # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
        x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights) + bias     # b, N, dim_out
        x_gconv0 = x_gconv
        for i in range(self.gcn_layer-1):
            x = x_gconv
            weights2 = torch.einsum('nd,dkio->nkio', node_embeddings, self.weights_pool2)  # N, cheb_k, dim_in, dim_out
            bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out

            #drop
            drop = F.relu(torch.rand((node_num, node_num), out=None) - self.droprate * torch.ones(node_num, node_num))
            drop = drop.to(supports.device)

            supports = torch.mul(supports, drop)

            x_g = torch.einsum("knm,bmc->bknc", supports, x)  # B, cheb_k, N, dim_in
            x_g = x_g.permute(0, 2, 1, 3)  # B, N, cheb_k, dim_in
            x_gconv = torch.einsum('bnki,nkio->bno', x_g, weights2) + bias + x_gconv0


        return x_gconv


class TGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim, gcn_layers, alpha, droprate):
        super(TGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = GCN(dim_in+self.hidden_dim, 2*dim_out, cheb_k, embed_dim, gcn_layers, alpha, droprate)
        self.update = GCN(dim_in+self.hidden_dim, dim_out, cheb_k, embed_dim, gcn_layers, alpha, droprate)

    def forward(self, x, state, node_embeddings):
        # x: B, num_nodes, input_dim
        # state: B, num_nodes, hidden_dim
        state = state.to(x.device)
        input_and_state = torch.cat((x, state), dim=-1)
        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))
        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z*state), dim=-1)
        hc = torch.tanh(self.update(candidate, node_embeddings))
        h = r*state + (1-r)*hc
        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)

class TimeGraphRegulation(nn.Module):
    def __init__(self, outfea, d, time_layer, time_drop):
        super(TimeGraphRegulation, self).__init__()
        self.vff = nn.Linear(outfea, outfea)
        self.ln = nn.LayerNorm(outfea)
        self.lnff = nn.LayerNorm(outfea)
        self.ff = nn.Sequential(
            nn.Linear(outfea, outfea),
            nn.ReLU(),
            nn.Linear(outfea, outfea)
        )
        self.d = d
        self.time_layer = time_layer
        self.time_drop = time_drop
    def forward(self, x, time_embeddings):
        supports = F.relu(torch.mm(time_embeddings, time_embeddings.transpose(0, 1)))
        supports = F.softmax(supports)
        eye = torch.eye(12).to(supports.device)
        supports = supports + eye
        value = self.vff(x)
        value = torch.cat(torch.split(value, self.d, -1), 0).permute(0, 2, 1, 3)

        mask = torch.tril(torch.ones(12, 12), diagonal=0).to(supports.device)
        supports = torch.mul(supports, mask)
        value = torch.matmul(supports, value)
        value0 = value

        for i in range(self.time_layer-1):
            drop = F.relu(torch.rand((12, 12), out=None) - self.time_drop * torch.ones(12, 12))
            drop = drop.to(supports.device)
            supports = torch.mul(supports, drop)
            A = supports
            value = torch.matmul(A, value) + value0
        value = torch.cat(torch.split(value, x.shape[0], 0), -1).permute(0, 2, 1, 3)
        value += x
        value = self.ln(value)
        x = self.ff(value) + value
        return self.lnff(x)
class FullTGCRN(nn.Module):
    def __init__(self, config):
        super(FullTGCRN, self).__init__()
        self.num_nodes = config['num_nodes']
        self.feature_dim = config['feature_dim']
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)
        self.num_layers = config.get('num_layers', 1)
        self.cheb_k = config.get('cheb_order', 2)
        assert self.num_layers >= 1, 'At least one DCRNN layer in the Encoder.'
        self.gcn_layers = config.get('gcn_layers')
        self.time_layers = config.get('time_layers', 2)
        self.alpha = config.get('alpha', 0.6)
        self.droprate = config.get('droprate', 0.4)
        self.time_drop = config.get('time_drop', 0.4)
        self.dcrnn_cells = nn.ModuleList()
        self.dcrnn_cells.append(TGCRNCell(self.num_nodes, self.feature_dim,
                                          self.hidden_dim, self.cheb_k, self.embed_dim, self.gcn_layers, self.alpha, self.droprate))
        for _ in range(1, self.num_layers):
            self.dcrnn_cells.append(AGCRNCell(self.num_nodes, self.hidden_dim,
                                              self.hidden_dim, self.cheb_k, self.embed_dim, self.gcn_layers, self.alpha, self.droprate))
        self.TimeGraph = TimeGraphRegulation(64, 64, self.time_layers, self.time_drop)

    def forward(self, x, init_state, node_embeddings, time_embeddings):
        # shape of x: (B, T, N, D)
        # shape of init_state: (num_layers, B, N, hidden_dim)
        assert x.shape[2] == self.num_nodes and x.shape[3] == self.feature_dim
        seq_length = x.shape[1]
        current_inputs = x
        output_hidden = []
        for i in range(self.num_layers):
            state = init_state[i]
            inner_states = []
            for t in range(seq_length):
                state = self.dcrnn_cells[i](current_inputs[:, t, :, :], state, node_embeddings)
                inner_states.append(state)
            output_hidden.append(state)
            current_inputs = torch.stack(inner_states, dim=1)
        current_inputs = self.TimeGraph(current_inputs, time_embeddings)
        # current_inputs: the outputs of last layer: (B, T, N, hidden_dim)
        # output_hidden: the last state for each layer: (num_layers, B, N, hidden_dim)
        # last_state: (B, N, hidden_dim)
        return current_inputs, output_hidden

    def init_hidden(self, batch_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.dcrnn_cells[i].init_hidden_state(batch_size))
        return torch.stack(init_states, dim=0)      # (num_layers, B, N, hidden_dim)


class AGCRN(AbstractTrafficStateModel):
    def __init__(self, config, data_feature):
        self.num_nodes = data_feature.get('num_nodes', 1)
        self.feature_dim = data_feature.get('feature_dim', 1)
        config['num_nodes'] = self.num_nodes
        config['feature_dim'] = self.feature_dim

        super().__init__(config, data_feature)
        self.input_window = config.get('input_window', 1)
        self.output_window = config.get('output_window', 1)
        self.output_dim = self.data_feature.get('output_dim', 1)
        self.hidden_dim = config.get('rnn_units', 64)
        self.embed_dim = config.get('embed_dim', 10)
        self.time_dim = config.get('time_dim', 10)
        self.node_embeddings = nn.Parameter(torch.randn(self.num_nodes, self.embed_dim), requires_grad=True)
        self.time_embeddings = nn.Parameter(torch.randn(12, self.time_dim), requires_grad=True)
        self.encoder = FullTGCRN(config)
        self.end_conv = nn.Conv2d(1, self.output_window * self.output_dim, kernel_size=(1, self.hidden_dim), bias=True)

        self.device = config.get('device', torch.device('cpu'))
        self._logger = getLogger()
        self._scaler = self.data_feature.get('scaler')
        self._init_parameters()

    def _init_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
            else:
                nn.init.uniform_(p)

    def forward(self, batch):
        # source: B, T_1, N, D
        # target: B, T_2, N, D
        # supports = F.softmax(F.relu(torch.mm(self.nodevec1, self.nodevec1.transpose(0,1))), dim=1)
        source = batch['X']

        init_state = self.encoder.init_hidden(source.shape[0])
        output, _ = self.encoder(source, init_state, self.node_embeddings, self.time_embeddings)  # B, T, N, hidden
        output = output[:, -1:, :, :]                                       # B, 1, N, hidden

        # CNN based predictor
        output = self.end_conv(output)                           # B, T*C, N, 1
        output = output.squeeze(-1).reshape(-1, self.output_window, self.output_dim, self.num_nodes)
        output = output.permute(0, 1, 3, 2)                      # B, T, N, C
        return output

    def calculate_loss(self, batch):
        y_true = batch['y']
        y_predicted = self.predict(batch)
        y_true = self._scaler.inverse_transform(y_true[..., :self.output_dim])
        y_predicted = self._scaler.inverse_transform(y_predicted[..., :self.output_dim])
        return loss.masked_mae_torch(y_predicted, y_true, 0)

    def predict(self, batch):
        return self.forward(batch)