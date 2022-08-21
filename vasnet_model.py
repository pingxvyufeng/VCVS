#encoding:utf-8


import torch
import torch.nn as nn
import torch.nn.functional as F
from config import  *
from layer_norm import  *
import numpy as np
from capsule import CapsuleLayer


# self attention
class SelfAttention(nn.Module):

    def __init__(self, apperture=-1, ignore_itself=False, input_size=1024, output_size=1024):
        super(SelfAttention, self).__init__()

        self.apperture = apperture
        self.ignore_itself = ignore_itself

        self.m = input_size
        self.output_size = output_size

        self.K = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.Q = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.V = nn.Linear(in_features=self.m, out_features=self.output_size, bias=False)
        self.output_linear = nn.Linear(in_features=self.output_size, out_features=self.m, bias=False)

        self.drop50 = nn.Dropout(0.1)



    def forward(self, x):
        n = x.shape[0]  # sequence length

        K = self.K(x)  # ENC (n x m) => (n x H) H= hidden size
        Q = self.Q(x)  # ENC (n x m) => (n x H) H= hidden size
        V = self.V(x)
        Q  = 0.06 * Q



        logits = torch.matmul(Q, K.transpose(1,0))

        if self.ignore_itself:
            # Zero the diagonal activations (a distance of each frame with itself)
            logits[torch.eye(n).byte()] = -float("Inf")

        if self.apperture > 0:
            # Set attention to zero to frames further than +/- apperture from the current one
            onesmask = torch.ones(n, n)
            trimask = torch.tril(onesmask, -self.apperture) + torch.triu(onesmask, self.apperture)
            logits[trimask == 1] = -float("Inf")

        att_weights_ = nn.functional.softmax(logits, dim=-1)
        weights = self.drop50(att_weights_)
        y = torch.matmul(V.transpose(1,0), weights).transpose(1,0)
        y = self.output_linear(y)

        return y, att_weights_



class MLP(nn.Module):
    def __init__(self,input_size,output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 1024)
        self.fc2 = nn.Linear(1024, output_size)
        #self.fc3 = nn.Linear(256, output_size)

    def forward(self, x):
        m = 1024
        din = x.view(-1, m)
        #print (din.size())
        dout = torch.tanh(self.fc1(din))

        dout = torch.tanh(self.fc2(dout))


        #dout = nn.functional.tanh(self.fc3(dout))


        return dout


class VASNet(nn.Module):

    def __init__(self):
        super(VASNet, self).__init__()

        self.m = 1024 # cnn features size
        self.hidden_size = 1024

        self.att = SelfAttention(input_size=self.m, output_size=self.m)

        self.mlp = MLP(input_size=1024,output_size=1024)
 
        self.ka = nn.Linear(in_features=self.m, out_features=1024)
        self.lstm = nn.GRU(input_size=1024, hidden_size=512, num_layers= 1, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(input_size=1024, hidden_size=1024, num_layers=1, batch_first=True, bidirectional=False)
        self.kb = nn.Linear(in_features=self.ka.out_features, out_features=1024)
        self.kc = nn.Linear(in_features=self.kb.out_features, out_features=1024)
        self.kd = nn.Linear(in_features=self.ka.out_features, out_features=1)

        self.sig = nn.Sigmoid()
        self.tanh = nn.Tanh()
        self.relu = nn.ReLU()
        self.drop50 = nn.Dropout(0.1)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)
        self.layer_norm_y = LayerNorm(self.m)
        self.layer_norm_ka = LayerNorm(self.ka.out_features)
        self.layer_norm_kb = LayerNorm(self.kb.out_features)
        self.layer_norm_kc = LayerNorm(self.kc.out_features)
        # capsnet
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=9, stride=1)
        self.primary_capsules = CapsuleLayer(num_capsules=1, num_route_nodes=-1, in_channels=32, out_channels=8,
                                             kernel_size=9, stride=2)
        self.digit_capsules = CapsuleLayer(num_capsules=1, num_route_nodes=8 * 8 * 8, in_channels=3,
                                           out_channels=16)



    def forward(self, x_list, seq_len):
        x_ls = []
        att_weights = []
        for i in range(len(x_list)):
            x = x_list[i].view(-1, x_list[i].shape[2]).unsqueeze(1)
            x, _ = self.lstm(x)
            x = x.squeeze(1)
            s, att_weight_ = self.att(x)
            s = s + x
            s = s.unsqueeze(0)
            s, _ = self.gru(s)
            p = self.layer_norm_y(s)
            p = p.squeeze(0)
            p = p[:seq_len, :]
            p = p.reshape(seq_len, 32, 32)
            x_ls.append(p)
            att_weights.append(att_weight_)
        caps_ls = []
        for a in range(len(x_ls)):
            x = x_ls[a].unsqueeze(1)
            x = F.relu(self.conv1(x), inplace=True)
            x = self.primary_capsules(x)
            caps_ls.append(x)
        outputs = torch.cat(caps_ls, dim=-1)
        outputs = self.digit_capsules(outputs).squeeze(2).transpose(0, 1)
        classes = (outputs ** 2).sum(dim=-1) ** 0.5
        classes = classes.squeeze(1).transpose(0, 1)
        att_weight = sum(att_weights)/len(att_weights)
        return classes, att_weight

        return y, att_weights_


if __name__ == "__main__":
    pass


