import math
import torch
import torch.nn as nn 
import torch.nn.functional as F 
import torch.nn.init as init
import numpy as np
from torch.autograd import Variable
import math
import sys
import os
import copy
from models.augSSL import sim_global


class GCN1(nn.Module):
    def __init__(self, in_features, out_features, dropout, alpha, hop = 1):
        super(GCN1, self).__init__()
        self.in_features = in_features
        self.hop = hop
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.w_lot = nn.ModuleList()
    
        for i in range(hop):
            in_features = (self.in_features) if(i==0) else out_features 
            self.w_lot.append(nn.Linear(in_features, out_features, bias=True))
       
    def forward(self, h_c, adj):
       
        b,c,n,t=h_c.shape
      
        h_c=h_c.reshape(-1,n,c)
        # adj normalize
        adj_rowsum = torch.sum(adj,dim=-1,keepdim=True)
        adj = adj.div(torch.where(adj_rowsum>1e-8, adj_rowsum, 1e-8*torch.ones(1,1).cuda())) # row normalize
        # weight aggregate
        
        for i in range(self.hop):
            # h_c = torch.matmul(adj,h_c)
            h_c=torch.einsum('kk,mkc->mkc',adj,h_c)
            h_c = self.leakyrelu(self.w_lot[i](h_c)) #(B, N, F)
            
        h_c=h_c.reshape(b,c,n,-1)
        return h_c
    

class SpatialHeteroModel(nn.Module):
    '''Spatial heterogeneity modeling by using a soft-clustering paradigm.
    '''
    def __init__(self, c_in, nmb_prototype, tau=0.5):
        super(SpatialHeteroModel, self).__init__()
        self.l2norm = lambda x: F.normalize(x, dim=1, p=2)
        self.prototypes = nn.Linear(c_in, nmb_prototype, bias=False)
        self.latent=nmb_prototype
        self.tau = tau
        self.d_model = c_in
        

        for m in self.modules():
            self.weights_init(m)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.fill_(0.0)

    def forward(self, z1, z2):
        """Compute the contrastive loss of batched data.
        :param z1, z2 (tensor): shape nlvc
        :param loss: contrastive loss
        """
        with torch.no_grad():
            w = self.prototypes.weight.data.clone()
            w = self.l2norm(w)
            self.prototypes.weight.copy_(w)
        
       
        b,c,n,t=z1.size()
        # z1=z1.narrow(3,1,1).squeeze()
        # z2=z2.narrow(3,1,1).squeeze()
        zc1 = self.prototypes(self.l2norm(z1.reshape(-1, self.d_model))) # nd -> nk, assignment q, embedding z
        zc2 = self.prototypes(self.l2norm(z2.reshape(-1, self.d_model))) # nd -> nk
        # zc1 = self.prototypes(z1.reshape(-1, self.d_model)) # nd -> nk, assignment q, embedding z
        # zc2 = self.prototypes(z2.reshape(-1, self.d_model))
        with torch.no_grad():
            q1 = sinkhorn(zc1.detach())
            q2 = sinkhorn(zc2.detach())
        # q1=F.softmax(zc1,dim=-1)
        # q2=F.softmax(zc2,dim=-1)
        l1 = - torch.mean(torch.sum(q1 * F.log_softmax(zc2 / self.tau, dim=1), dim=1))
        l2 = - torch.mean(torch.sum(q2 * F.log_softmax(zc1 / self.tau, dim=1), dim=1))
        # l1=l2=0
        # q1=q1.reshape(b,n,-1)
      
        clu=zc1.reshape(b,n,-1)
        # clu=torch.softmax(clu/self.tau,dim=-1)
        
        
        return clu, l1+l2
    
@torch.no_grad()
def sinkhorn(out, epsilon=0.05, sinkhorn_iterations=3):
    Q = torch.exp(out / epsilon).t() # Q is K-by-B for consistency with notations from our paper
    B = Q.shape[1] # number of samples to assign
    K = Q.shape[0] # how many prototypes
    
    # make the matrix sums to 1
    sum_Q = torch.sum(Q)
    Q /= sum_Q
    
    for it in range(sinkhorn_iterations):
        # normalize each row: total weight per prototype must be 1/K
        Q /= torch.sum(Q, dim=1, keepdim=True)
        Q /= K
        
        # normalize each column: total weight per sample must be 1/B
        Q /= torch.sum(Q, dim=0, keepdim=True)
        Q /= B
    
    Q *= B # the colomns must sum to 1 so that Q is an assignment
    return Q.t()


# 检查矩阵是否对称

 


def aug_topology(sim_mx, input_graph, percent=0.2):
    """Generate the data augumentation from topology (graph structure) perspective 
        for undirected graph without self-loop.
    :param sim_mx: tensor, symmetric similarity, [v,v]
    :param input_graph: tensor, adjacency matrix without self-loop, [v,v]
    :return aug_graph: tensor, augmented adjacency matrix on cuda, [v,v]
    """    
    ## edge dropping starts here
    drop_percent = percent / 2
    
    input_graph = torch.where(input_graph >0, 1, input_graph)

    print(torch.all(torch.eq(input_graph, input_graph.t())))

    index_list = input_graph.nonzero() # list of edges [row_idx, col_idx]
    


    # edge_num = int((index_list.shape[0]-input_graph.shape[0])/2)  # treat one undirected edge as two edges  beijing
    # edge_num = int(index_list.shape[0]/2)   #xian
    edge_num=739

    edge_mask = (input_graph>0).tril(diagonal=-1)
    
    add_drop_num = int(edge_num * drop_percent) 
    aug_graph = copy.deepcopy(input_graph) 

    drop_prob = torch.softmax(sim_mx[edge_mask], dim=0)
    drop_prob = (1. - drop_prob).cpu().detach().numpy() # normalized similarity to get sampling probability 
    drop_prob /= drop_prob.sum()
   
    drop_list = np.random.choice(edge_num, size=add_drop_num, p=drop_prob)
    # drop_list = np.random.choice(edge_num, size=add_drop_num)
    drop_index = index_list[drop_list]
    
    zeros = torch.zeros_like(aug_graph[0, 0])
    aug_graph[drop_index[:, 0], drop_index[:, 1]] = zeros
    aug_graph[drop_index[:, 1], drop_index[:, 0]] = zeros

    ## edge adding starts here
    node_num = input_graph.shape[0]
    x, y = np.meshgrid(range(node_num), range(node_num), indexing='ij')
    mask = y < x
    x, y = x[mask], y[mask]

    add_prob = sim_mx[torch.ones(sim_mx.size(), dtype=bool).tril(diagonal=-1)] # .numpy()
    add_prob = torch.softmax(add_prob, dim=0).cpu().detach().numpy()
    add_list = np.random.choice(int((node_num * node_num - node_num) / 2), 
                                size=add_drop_num, p=add_prob)
    # add_list = np.random.choice(int((node_num * node_num - node_num) / 2), 
    #                             size=add_drop_num)
    
    ones = torch.ones_like(aug_graph[0, 0])
    aug_graph[x[add_list], y[add_list]] = ones
    aug_graph[y[add_list], x[add_list]] = ones
    
    return aug_graph   


def aug_traffic(t_sim_mx, flow_data, percent=0.1):
    """Generate the data augumentation from traffic (node attribute) perspective.
    :param t_sim_mx: temporal similarity matrix after softmax, [l,n,v]
    :param flow_data: input flow data, [n,l,v,c]
    """
    l, n, v = t_sim_mx.shape
    mask_num = int(n * l * v * percent)
    aug_flow = flow_data
    aug_flow=aug_flow.permute(0,3,2,1)
    mask_prob = (1. - t_sim_mx.permute(1, 0, 2).reshape(-1)).cpu().detach().numpy()
    mask_prob /= mask_prob.sum()

    x, y, z = np.meshgrid(range(n), range(l), range(v), indexing='ij')
    mask_list = np.random.choice(n * l * v, size=mask_num, p=mask_prob)
    # mask_list = np.random.choice(n * l * v, size=mask_num)

    zeros = torch.zeros_like(aug_flow[0, 0, 0])
    aug_flow[
        x.reshape(-1)[mask_list], 
        y.reshape(-1)[mask_list], 
        z.reshape(-1)[mask_list]] = zeros 
    aug_flow=aug_flow.permute(0,3,2,1)
    return aug_flow

class TemporalConvLayer(nn.Module):
    def __init__(self, kt, c_in, c_out, act="relu"):
        super(TemporalConvLayer, self).__init__()
        self.kt = kt
        self.act = act
        self.c_out = c_out
        self.align = Align(c_in, c_out)
        if self.act == "GLU":
            self.conv = nn.Conv2d(c_in, c_out * 2, (kt, 1), 1)
        else:
            self.conv = nn.Conv2d(c_in, c_out, (kt, 1), 1)

    def forward(self, x):
        """
        :param x: (n,c,l,v)
        :return: (n,c,l-kt+1,v)
        """
        x=x.permute(0,1,3,2)
        x_in = self.align(x)[:, :, self.kt - 1:, :]  
        if self.act == "GLU":
            x_conv = self.conv(x)
            return (x_conv[:, :self.c_out, :, :] + x_in) * torch.sigmoid(x_conv[:, self.c_out:, :, :])
        if self.act == "sigmoid":
            return torch.sigmoid(self.conv(x) + x_in)  
        return torch.relu(self.conv(x) + x_in)  
    
class Pooler(nn.Module):
    '''Pooling the token representations of region time series into the region level.
    '''
    def __init__(self, n_query, d_model, agg='avg'):
        """
        :param n_query: number of query
        :param d_model: dimension of model 
        """
        super(Pooler, self).__init__()

        ## attention matirx
        self.att = nn.Conv2d(d_model, n_query,1) 
        self.align = Align(d_model, d_model)
        self.softmax = nn.Softmax(dim=2) # softmax on the seq_length dim, nclv

        self.d_model = d_model
        self.n_query = n_query 
        if agg == 'avg':
            self.agg = nn.AvgPool2d(kernel_size=(n_query, 1), stride=1)
        elif agg == 'max':
            self.agg = nn.MaxPool2d(kernel_size=(n_query, 1), stride=1)
        else:
            raise ValueError('Pooler supports [avg, max]')
        
    def forward(self, x):
        """
        :param x: key sequence of region embeding, nclv
        :return x: hidden embedding used for conv, ncqv
        :return x_agg: region embedding for spatial similarity, nvc
        :return A: temporal attention, lnv
        """
        x_in = self.align(x)[:, :, -self.n_query:, :] # ncqv
        # calculate the attention matrix A using key x  
        A = self.att(x) # x: nclv, A: nqlv 
        A = F.softmax(A, dim=2) # nqlv

        # calculate region embeding using attention matrix A
        x = torch.einsum('nclv,nqlv->ncqv', x, A)
        x_agg = self.agg(x).squeeze(2) # ncqv->ncv
        x_agg = torch.einsum('ncv->nvc', x_agg) # ncv->nvc

        # calculate the temporal simlarity (prob)
        A = torch.einsum('nqlv->lnqv', A)
        A = self.softmax(self.agg(A).squeeze(2)) # A: lnqv->lnv
        # print(x.shape)
        # print(x_in.shape)
        return torch.relu(x + x_in), x_agg, A



class Align(nn.Module):
    def __init__(self, c_in, c_out):
        '''Align the input and output.
        '''
        super(Align, self).__init__()
        self.c_in = c_in
        self.c_out = c_out
        if c_in > c_out:
            self.conv1x1 = nn.Conv2d(c_in, c_out, 1)  # filter=(1,1), similar to fc

    def forward(self, x):  # x: (n,c,l,v)
        if self.c_in > self.c_out:
            return self.conv1x1(x)
        if self.c_in < self.c_out:
            return F.pad(x, [0, 0, 0, 0, 0, self.c_out - self.c_in, 0, 0])
        return x  














class SSL_cluster(nn.Module):
    def __init__(self, in_features, out_features, t_len, dropout, alpha, latend_num, gcn_hop):
        super(SSL_cluster, self).__init__()
        self.in_features = in_features
        self.out_features=out_features
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        self.gcn_1=GCN1(in_features=out_features, out_features=out_features, \
                                       dropout=dropout, alpha=alpha, hop = gcn_hop)
        self.shm=SpatialHeteroModel(c_in=out_features*t_len, nmb_prototype= latend_num, tau=0.5)
       
        self.start_conv = nn.Conv2d(in_channels=in_features,
                                    out_channels=out_features,
                                    kernel_size=(1,1))
        self.start_conv_1 = nn.Conv2d(in_channels=in_features,
                                    out_channels=out_features,
                                    kernel_size=(1,1))
   
    def apply_bn(self, x):
        # Batch normalization of 3D tensor x
        bn_module = nn.BatchNorm1d(x.size()[1]).cuda(1)
        x = bn_module(x)
        return x

    def forward(self,x,x_aug,adj,adj_aug):
        """
        :param X_lots: Concat of the outputs of CxtConv and PA_approximation (batch_size, N, in_features).  
        :param adj: adj_merge (N, N). 
        :return: Output soft clustering representation for each parking lot of shape (batch_size, N, out_features).
        """
                               
        b,c,n,t=x.size()
        x=self.start_conv(x) 
        x_aug=self.start_conv(x_aug)
       
        z1=self.gcn_1(x,adj)
        z2=self.gcn_1(x,adj_aug)
        
       
        A,s_loss=self.shm(z1,z2)
      
       
        return s_loss,A,z1
