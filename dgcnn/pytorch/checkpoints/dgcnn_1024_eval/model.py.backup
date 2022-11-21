#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
"""


import os
import sys
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

#knn分类算法
'''knn分类算法： 近朱者赤，近墨者黑，根据新样本的k个最近邻居来判定其类别  '''
''' 从已分类的所有点中选出k个离未标记点最近的点，选出的k个点中标签出现频率最高的就是未标记点的类别 '''
def knn(x, k):
    inner = -2*torch.matmul(x.transpose(2, 1), x)  #两个张量的矩阵乘积 0维张量是标量 1维张量是向量 2维张量是矩阵  transpose 转置函数 2号轴与1号轴互换  在这里是把k与num_points换一个位置
    xx = torch.sum(x**2, dim=1, keepdim=True)
    pairwise_distance = -xx - inner - xx.transpose(2, 1)
 
    idx = pairwise_distance.topk(k=k, dim=-1)[1]   # (batch_size, num_points, k)
    return idx

#获取图特征函数
def get_graph_feature(x, k=20, idx=None):
    batch_size = x.size(0)
    num_points = x.size(2)
    x = x.view(batch_size, -1, num_points) #重构x为 结构为(batch_size,适配,num_points)形式的三维张量
    if idx is None:
        idx = knn(x, k=k)   # (batch_size, num_points, k)   batch_size,num_points,k 组成的3维张量数组  batch_size个二维数组构成的三维数字；每个二维数字由num_points个k维向量构成
    device = torch.device('cuda')

    idx_base = torch.arange(0, batch_size, device=device).view(-1, 1, 1)*num_points

    idx = idx + idx_base

    idx = idx.view(-1)
 
    _, num_dims, _ = x.size()

    x = x.transpose(2, 1).contiguous()   # (batch_size, num_points, num_dims)  -> (batch_size*num_points, num_dims) #   batch_size * num_points * k + range(0, batch_size*num_points)
    feature = x.view(batch_size*num_points, -1)[idx, :]
    feature = feature.view(batch_size, num_points, k, num_dims) 
    x = x.view(batch_size, num_points, 1, num_dims).repeat(1, 1, k, 1)
    
    feature = torch.cat((feature-x, x), dim=3).permute(0, 3, 1, 2).contiguous()
  
    return feature

#PointNet 点云网络模型
class PointNet(nn.Module):
    ''' 注意 作者没有去实现原版点云网络的 T-net 结构，事实上相当一部分点云网络的应用发现删去T-net会使之效果更好'''
    def __init__(self, args, output_channels=40):
        super(PointNet, self).__init__()
        self.args = args
        self.conv1 = nn.Conv1d(3, 64, kernel_size=1, bias=False)
        self.conv2 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv3 = nn.Conv1d(64, 64, kernel_size=1, bias=False)
        self.conv4 = nn.Conv1d(64, 128, kernel_size=1, bias=False)
        self.conv5 = nn.Conv1d(128, args.emb_dims, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)
        self.linear1 = nn.Linear(args.emb_dims, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout()
        self.linear2 = nn.Linear(512, output_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.adaptive_max_pool1d(x, 1).squeeze()
        x = F.relu(self.bn6(self.linear1(x)))
        x = self.dp1(x)
        x = self.linear2(x)
        return x

#DGCNN 动态图卷积神经网络模型
class DGCNN(nn.Module):
    def __init__(self, args, output_channels=40):
        super(DGCNN, self).__init__()
        self.args = args
        self.k = args.k
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(64)
        self.bn3 = nn.BatchNorm2d(128)
        self.bn4 = nn.BatchNorm2d(256)
        self.bn5 = nn.BatchNorm1d(args.emb_dims)

        self.conv1 = nn.Sequential(nn.Conv2d(6, 64, kernel_size=1, bias=False),
                                   self.bn1,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv2 = nn.Sequential(nn.Conv2d(64*2, 64, kernel_size=1, bias=False),
                                   self.bn2,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv3 = nn.Sequential(nn.Conv2d(64*2, 128, kernel_size=1, bias=False),
                                   self.bn3,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv4 = nn.Sequential(nn.Conv2d(128*2, 256, kernel_size=1, bias=False),
                                   self.bn4,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.conv5 = nn.Sequential(nn.Conv1d(512, args.emb_dims, kernel_size=1, bias=False),
                                   self.bn5,
                                   nn.LeakyReLU(negative_slope=0.2))
        self.linear1 = nn.Linear(args.emb_dims*2, 512, bias=False)
        self.bn6 = nn.BatchNorm1d(512)
        self.dp1 = nn.Dropout(p=args.dropout)
        self.linear2 = nn.Linear(512, 256)
        self.bn7 = nn.BatchNorm1d(256)
        self.dp2 = nn.Dropout(p=args.dropout)
        self.linear3 = nn.Linear(256, output_channels)

    def forward(self, x):
        batch_size = x.size(0)

        x = get_graph_feature(x, k=self.k)      #torch.Size([16, 3, 1024]) ->  torch.Size([16, 6, 1024, 20]) 获取特征 n*3 n*k*6
        x = self.conv1(x)    #torch.Size([16, 6, 1024, 20]) -> torch.Size([16, 64, 1024, 20])
        x1 = x.max(dim=-1, keepdim=False)[0]

        x = get_graph_feature(x1, k=self.k)  # torch.Size([16, 64, 1024, 20])  ->  torch.Size([16, 128, 1024, 20])
        x = self.conv2(x)     #torch.Size([16, 128, 1024, 20])  -> torch.Size([16, 64, 1024, 20])
        x2 = x.max(dim=-1, keepdim=False)[0]


        x = get_graph_feature(x2, k=self.k)  #torch.Size([16, 128, 1024, 20])
        x = self.conv3(x)   #torch.Size([16, 128, 1024, 20])
        x3 = x.max(dim=-1, keepdim=False)[0]


        x = get_graph_feature(x3, k=self.k)  #torch.Size([16, 256, 1024, 20])
        x = self.conv4(x)  # torch.Size([16, 256, 1024, 20])
        x4 = x.max(dim=-1, keepdim=False)[0]


        x = torch.cat((x1, x2, x3, x4), dim=1)  #torch.Size([16, 512, 1024])


        x = self.conv5(x)   #torch.Size([16, 1024, 1024])
        x1 = F.adaptive_max_pool1d(x, 1).view(batch_size, -1)
        x2 = F.adaptive_avg_pool1d(x, 1).view(batch_size, -1)

        x = torch.cat((x1, x2), 1)   # torch.Size([16, 2048])


        x = F.leaky_relu(self.bn6(self.linear1(x)), negative_slope=0.2)  #torch.Size([16, 512])
        x = self.dp1(x) # torch.Size([16, 512])
        x = F.leaky_relu(self.bn7(self.linear2(x)), negative_slope=0.2) # torch.Size([16, 256])
        x = self.dp2(x) # torch.Size([16, 256])
        x = self.linear3(x) #torch.Size([16, 40])

        return x
