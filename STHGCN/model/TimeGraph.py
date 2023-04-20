import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.decomposition import PCA
class TimeGCN(nn.Module):
    def __init__(self, dim_in, dim_out, cheb_k, t, gcn_layers, droprate):
        super(TimeGCN, self).__init__()
        self.cheb_k = cheb_k
        self.weights = nn.Parameter(torch.FloatTensor(t, cheb_k,dim_in, dim_out))

        self.pool_num = nn.Parameter(torch.FloatTensor(12,t))
        self.pool_graph = nn.Parameter(torch.FloatTensor(12,2))
        self.gate = nn.Parameter(torch.ones(11,1,1))
        self.bias = nn.Parameter(torch.FloatTensor(12, dim_out))
        self.gcn_layer = gcn_layers
        self.g = nn.Parameter(torch.FloatTensor(24, 1))
        self.droprate = droprate


    def forward(self, x, time_embeddings):

        eyes = torch.load("eye.pt").to(x.device)
        trans = torch.load("transdown.pt").to(x.device)
        su = torch.mul(self.gate,trans)+eyes
        su = F.relu(su)
        f = torch.eye(12).to(x.device)
        for i in range(11):
            f = torch.matmul(f,su[i,:,:])
        supports = f
        #print(f)
        s = []
        weights = torch.einsum("tn,nkio->tkio",self.pool_num,self.weights)
        time_num = x.shape[1]
        T_supports = F.softmax(F.relu(torch.mm(time_embeddings, time_embeddings.transpose(0, 1))), dim=1)
        T_supports = T_supports.to(x.device)
        #supports = torch.load("support.pt")

        supports = supports.to(x.device)
        support_set = [torch.eye(time_num).to(supports.device), supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * supports, support_set[-1]) + support_set[-2])
        supports1 = torch.stack(support_set, dim=0)

        s.append(supports1)
        candidate = torch.cat((supports, T_supports), dim=-1)
        g1 = torch.matmul(candidate, self.g)
        g2 = 1-g1
        g = torch.cat((g1, g2), dim=1)
        
        weights = torch.einsum("ts,tkio->tksio", g, weights)
        support_set = [torch.eye(time_num).to(supports.device), T_supports]
        # default cheb_k = 3
        for k in range(2, self.cheb_k):
            support_set.append(torch.matmul(2 * T_supports, support_set[-1]) + support_set[-2])
        supports2 = torch.stack(support_set, dim=0)
        s.append(supports2)
        supports = torch.stack(s, dim=0)

        #print(supports.shape)
          # N, cheb_k, dim_in, dim_out
         # N, dim_out
        x_g = torch.einsum("sktp,btnc->bspknc", supports, x)  # B,s, cheb_k, N, dim_in
        x_g = x_g.permute(0, 1, 2,4, 3, 5)  # B, N, cheb_k, dim_in

        x_gconv = torch.einsum('bstnki,tksio->btno', x_g, weights)   # b, N, dim_out
        x_gconv = x_gconv.permute(0, 2, 1, 3)
        x_gconv =x_gconv+self.bias
        x_gconv = x_gconv.permute(0, 2, 1, 3)


        return x_gconv


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
        self.pad = nn.ZeroPad2d(padding=(0,1,1,0))
        self.GCN = TimeGCN(64,64,2,3,2,0.4)
    def forward(self, x, time_embeddings):
        # a1 = torch.eye(11)
        #
        # ai = self.pad(a1)


        x1 = self.vff(x)
        #g_x = torch.einsum('tp,btnd->bpnd', supports, x1)

        g_x = self.GCN(x1,time_embeddings)
        g_x = self.GCN(g_x, time_embeddings) + x1 +g_x
        g_x = self.vff(g_x)+x1
        # print(g_x.shape)
        # print(x.shape)
        Y = self.vff(F.softmax(g_x)) + x1
        # print(Y.shape)


        #Z = self.vff(self.vff(Y))+Y
        Z = Y

        return Z