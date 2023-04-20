import torch
import torch.nn.functional as F
import torch.nn as nn


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
        node_num = 170

        self.ff = nn.Linear(2 * node_num, node_num)

    def common_loss(self,emb1, emb2):
        emb1 = emb1 - torch.mean(emb1, dim=0, keepdim=True)
        emb2 = emb2 - torch.mean(emb2, dim=0, keepdim=True)
        emb1 = torch.nn.functional.normalize(emb1, p=2, dim=1)
        emb2 = torch.nn.functional.normalize(emb2, p=2, dim=1)
        cov1 = torch.matmul(emb1, emb1.t())
        cov2 = torch.matmul(emb2, emb2.t())
        cost = torch.mean((cov1 - cov2) ** 2)
        return cost

    def loss_dependence(self,emb1, emb2, dim):
        R = torch.eye(dim).cuda() - (1 / dim) * torch.ones(dim, dim).cuda()
        K1 = torch.mm(emb1, emb1.t())
        K2 = torch.mm(emb2, emb2.t())
        RK1 = torch.mm(R, K1)
        RK2 = torch.mm(R, K2)
        HSIC = torch.trace(torch.mm(RK1, RK2))
        return HSIC
    def forward(self, x, node_embeddings, nodevec1, nodevec2):

        node_num = node_embeddings.shape[0]

        supports = F.tanh(torch.mm(node_embeddings, node_embeddings.transpose(0, 1))) + F.softmax(
            F.tanh(torch.mm(nodevec1, nodevec2)), dim=1)

        # adp = F.softmax(F.tanh(torch.mm(nodevec1, nodevec2)), dim=1)
        adp1 = F.softmax(F.tanh(torch.mm(nodevec1, nodevec1.transpose(0, 1))), dim=1)
        adp2 = F.tanh(torch.mm(node_embeddings, node_embeddings.transpose(0, 1)))
        can = torch.cat((adp1, adp2), dim=-1)
        gate = self.ff(can)
        supports = adp1 * gate + adp2 * (1 - gate)

        #similarity = torch.cosine_similarity(node_embeddings, nodevec1, dim=0)
        # similarity = F.softmax(F.relu(similarity))
        #common_loss = self.common_loss(node_embeddings,nodevec1)*20
        #print("commom",common_loss )
        loss_dependence = self.loss_dependence(node_embeddings,nodevec1,170)
        #print("loss_dependence ",loss_dependence )
        #similarity = torch.sum(similarity)
        #similarity = common_loss+loss_dependence
        #similarity =common_loss
        similarity =0
        support_set = [torch.eye(node_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) - support_set[-2])
        supports = torch.stack(support_set, dim=0)

        weights = torch.einsum('np,pkio->nkio', node_embeddings, self.weights_pool)  # N, cheb_k, dim_in, dim_out
        bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out

        x_g = torch.einsum("knm,btmc->btknc", supports, x)  # B, cheb_k, N, dim_in
        x_g = x_g.permute(0, 1, 3, 2, 4)  # B, N, cheb_k, dim_in

        x_gconv = torch.einsum('btnki,nkio->btno', -x_g, weights) + bias  # b, N, dim_out
        x_gconv0 = x_gconv

        for i in range(self.gcn_layer - 1):
            x = x_gconv
            weights2 = torch.einsum('np,pkio->nkio', node_embeddings, self.weights_pool2)  # N, cheb_k, dim_in, dim_out
            bias = torch.matmul(node_embeddings, self.bias_pool)  # N, dim_out
            # drop
            drop = F.relu(torch.rand((node_num, node_num), out=None) - self.droprate * torch.ones(node_num, node_num))
            drop = drop.to(supports.device)
            supports = torch.mul(supports, drop)
            x_g = torch.einsum("knm,btmc->btknc", supports, x)  # B, cheb_k, N, dim_in
            x_g = x_g.permute(0, 1, 3, 2, 4)  # B, N, cheb_k, dim_in

            x_gconv = torch.einsum('btnki,nkio->btno', x_g, weights2) + bias + x_gconv0
        return x_gconv, similarity