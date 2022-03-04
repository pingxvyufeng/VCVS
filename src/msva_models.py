import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

import config
from capsule import CapsuleLayer
max_seg_frames = 10




class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class SelfAttention(nn.Module):
    def __init__(self, apperture, ignore_itself=False, input_size=1024, output_size=1024,
                 dropout=0.5):  # apperture -1 to ignore
        super(SelfAttention, self).__init__()
        self.apperture = apperture
        self.ignore_itself = ignore_itself
        self.m = input_size
        self.output_size = output_size
        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        n = x.shape[0]
        K = self.K(x)
        Q = self.Q(x)
        V = self.V(x)
        Q *= 0.06
        logits = torch.matmul(Q, K.transpose(1, 0))
        if self.ignore_itself:
            logits[torch.eye(n).byte()] = -float("Inf")
        if self.apperture > 0:
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")
        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.dropout(att_weights_)
        y = torch.matmul(V.transpose(1, 0), weights).transpose(1, 0)
        y = self.output_linear(y)
        return y, att_weights_



class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=max_seg_frames, dim=1, alpha=1.0,
                 normalize_input=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        self.conv = nn.Conv2d(dim, num_clusters, kernel_size=(1, 1), bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        self._init_params()

    def _init_params(self):
        self.conv.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.conv.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, x):
        N, C = x.shape[:2]

        if self.normalize_input:
            x = F.normalize(x, p=2, dim=1)  # across descriptor dim

        # soft-assignment
        soft_assign = self.conv(x).view(N, self.num_clusters, -1)
        soft_assign = F.softmax(soft_assign, dim=1)

        x_flatten = x.view(N, C, -1)

        # calculate residuals to each clusters
        residual = x_flatten.expand(self.num_clusters, -1, -1, -1).permute(1, 0, 2, 3) - \
                   self.centroids.expand(x_flatten.size(-1), -1, -1).permute(1, 2, 0).unsqueeze(0)
        residual *= soft_assign.unsqueeze(2)
        vlad = residual.sum(dim=-1)

        vlad = F.normalize(vlad, p=2, dim=2)  # intra-normalization
        vlad = vlad.view(x.size(0), -1)  # flatten
        vlad = F.normalize(vlad, p=2, dim=1)  # L2 normalize

        return vlad




class MSVA_Gen_auto(nn.Module):  # MSVA auto variation train and learn for different shapes of N input features
    def __init__(self, feat_input, cmb):
        super(MSVA_Gen_auto, self).__init__()
        self.cmb = cmb
        self.feat_input = feat_input
        self.m = self.feat_input.feature_size  # cnn features size
        self.hidden_size = self.m
        self.apperture = self.feat_input.apperture
        self.att_1_3_size = self.feat_input.feature_size_1_3
        self.att1_3 = SelfAttention(apperture=self.apperture, input_size=self.att_1_3_size,
                                    output_size=self.att_1_3_size, dropout=self.feat_input.att_dropout1)
        self.ka1_3 = nn.Linear(in_features=self.att_1_3_size, out_features=self.feat_input.L1_out)
        self.kb = nn.Linear(in_features=self.ka1_3.out_features, out_features=self.feat_input.L2_out)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=self.feat_input.L3_out)
        self.kd = nn.Linear(in_features=self.kc.out_features, out_features=self.feat_input.pred_out)
        self.sig = nn.Sigmoid()
        self.relu = nn.ReLU()

        self.dropout = nn.Dropout(self.feat_input.dropout1)
        self.softmax = nn.Softmax(dim=0)
        self.layer_norm_y_1_3 = LayerNorm(self.att_1_3_size)
        self.layer_norm_y_4 = LayerNorm(self.att_1_3_size)
        self.layer_norm_kc = LayerNorm(self.kc.out_features)
        self.layer_norm_kd = LayerNorm(self.kd.out_features)
        #netvlad
        self.VLAD = NetVLAD(num_clusters=1, dim=130, alpha=1.0, normalize_input=True)
        #capsnet
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=1, num_route_nodes=-1, in_channels=32, out_channels=8,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=config.NUM_CLASSES, num_route_nodes=8 * 8 * 8, in_channels=3,
                                           out_channels=16)
        #seq2seq
        self.lstm = nn.GRU(input_size=1024, hidden_size=512, num_layers=1, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(input_size=1024, hidden_size=1024, num_layers=1, batch_first=True, bidirectional=False)


    def forward(self, x_list, seq_len):
        x_ls = []
        att_weights_ = []
        for i in range(len(x_list)):

            #只对深层特征和浅层特征进行自注意力机制的计算
            if i != 2:
                x = x_list[i].permute(3,1,2,0)
                x = self.VLAD(x).unsqueeze(0).permute(2,0,1)
                x,_ = self.lstm(x)
                x = x.squeeze(1)
                y, att_weights = self.att1_3(x)
                att_weights_ = att_weights
                y = y + x
                y = self.dropout(y)
                y = y.unsqueeze(0)
                y,_ = self.gru(y)
                y = self.layer_norm_y_1_3(y)
                y = y.squeeze(0)
                y = y[:seq_len,:]
                y = y.reshape(seq_len,32,32) # 展开（1,1024）特征为（32,32）
                x_ls.append(y)
            else :
                #对于光流特征无需太多处理，进行一定的标准化即可
                x = x_list[i].permute(3,1,2,0)
                x = self.VLAD(x).unsqueeze(0).permute(2, 0, 1)
                y = self.layer_norm_y_1_3(x)
                y = y.squeeze(0)
                y = y[:seq_len, :]
                y = y.reshape(seq_len, 32, 32)
                x_ls.append(y)
        caps_ls = []
        for a in range(len(x_ls)):
            x = x_ls[a].unsqueeze(1)
            x = F.relu(self.conv1(x), inplace=True)
            x = self.primary_capsules(x)
            caps_ls.append(x)
        outputs = torch.cat(caps_ls, dim=-1)
        outputs = self.digit_capsules(outputs).squeeze(2).transpose(0, 1)
        classes = (outputs ** 2).sum(dim=-1) ** 0.5
        classes = classes.squeeze(1).transpose(0,1)

        return classes,att_weights_





if __name__ == "__main__":
    pass
