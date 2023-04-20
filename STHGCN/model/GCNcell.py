import torch
import torch.nn as nn
from model.GCN import GCN

class TGCRNCell(nn.Module):
    def __init__(self, node_num, dim_in, dim_out, cheb_k, embed_dim,  gcn_layers, alpha, droprate):
        super(TGCRNCell, self).__init__()
        self.node_num = node_num
        self.hidden_dim = dim_out
        self.gate = GCN(dim_in, 2*dim_out,5, cheb_k, embed_dim, gcn_layers, alpha, droprate)
        self.update = GCN(dim_in, dim_out,5, cheb_k, embed_dim, gcn_layers, alpha, droprate)
        self.ln = nn.LayerNorm(dim_out)
        self.vff = nn.Linear(dim_in, dim_out)
    def forward(self, x, state, node_embeddings):
        #x: B, num_nodes, input_dim
        #state: B, num_nodes, hidden_dim


        input_and_state = x


        z_r = torch.sigmoid(self.gate(input_and_state, node_embeddings))

        z, r = torch.split(z_r, self.hidden_dim, dim=-1)
        candidate = torch.cat((x, z), dim=-1)

        hc = torch.tanh(self.update(candidate, node_embeddings))

        h = r*input_and_state + (1-r)*hc

        return h

    def init_hidden_state(self, batch_size):
        return torch.zeros(batch_size, self.node_num, self.hidden_dim)