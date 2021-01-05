# _*_ coding:utf-8 _*_
# __author: zhangxin
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model.BasicModule import BaseModule
from data.TrainDataLoader import TrainDataLoader
from sklearn.cluster import MeanShift ,estimate_bandwidth

class PTransD(BaseModule):

    def __init__(self,
                 ent_tot,
                 rel_tot,
                 dim_e=100,
                 dim_r=100,
                 p_norm=1,#得分用的一范
                 norm_flag=1,#实体向量需要单位化
                 margin=None,
                 k = 3,
                 epsilon=None,
                 batch_size = None,
                 regul_rate = None,
                 kl_rate = None):
        super(PTransD, self).__init__()
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.k = k
        self.margin = margin
        self.epsilon = epsilon
        self.p_norm = p_norm
        self.norm_flag = norm_flag
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.kl_rate = kl_rate
        self.batch_size = batch_size
        self.regul_rate = regul_rate


        #存储固定大小的词典的嵌入向量查找表
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_transfer = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_transfer = nn.Embedding(self.rel_tot, self.dim_r)
#        self.entity_probability_matrix = nn.Embedding(self.ent_tot,self.ent_tot)
#        self.entity_transfer_probability_matrix = nn.Embedding(self.ent_tot, self.ent_tot)

        #向量满足均匀分布U~(-a,a)
        if margin == None or epsilon == None:
            nn.init.xavier_uniform_(self.ent_embeddings.weight.data)
            nn.init.xavier_uniform_(self.rel_embeddings.weight.data)
            nn.init.xavier_uniform_(self.ent_transfer.weight.data)
            nn.init.xavier_uniform_(self.rel_transfer.weight.data)
        #向量满足均匀分布U~(a,b)，为什么这么设置我也不知道
        else:
            self.ent_embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim_e]), requires_grad=False
            )
            self.rel_embedding_range = nn.Parameter(
                torch.Tensor([(self.margin + self.epsilon) / self.dim_r]), requires_grad=False
            )
            nn.init.uniform_(
                tensor=self.ent_embeddings.weight.data,
                a=-self.ent_embedding_range.item(),
                b=self.ent_embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_embeddings.weight.data,
                a=-self.rel_embedding_range.item(),
                b=self.rel_embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.ent_transfer.weight.data,
                a=-self.ent_embedding_range.item(),
                b=self.ent_embedding_range.item()
            )
            nn.init.uniform_(
                tensor=self.rel_transfer.weight.data,
                a=-self.rel_embedding_range.item(),
                b=self.rel_embedding_range.item()
            )
        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False #不需计算梯度的变量
            self.margin_flag = True
        else:
            self.margin_flag = False

    #张量补零
    def _resize(self, tensor, axis, size):
        shape = tensor.size()
        osize = shape[axis]
        if osize == size:
            return tensor
        if (osize > size):
            return torch.narrow(tensor, axis, 0, size)
        paddings = []
        for i in range(len(shape)):
            if i == axis:
                paddings = [0, size - osize] + paddings
            else:
                paddings = [0, 0] + paddings
        return F.pad(tensor, paddings, mode="constant", value=0)

    def _calc(self, h, t, r):
        #将h,t,r进行标准化
        if self.norm_flag:
            h = F.normalize(h, 2, -1)
            r = F.normalize(r, 2, -1)
            t = F.normalize(t, 2, -1)
        score = h + (r - t)
        score = torch.norm(score, self.p_norm, -1).flatten() #求score的2范，并展开为一维Tensor
        return score

    def meanshift(self,matrix_transfer):  # 返回{transfer标号：对应的类}
        data = matrix_transfer.weight.data
        transfer_dict = {}
        e_mean_dict = {}

        bandwidth = estimate_bandwidth(data, quantile=0.2)
        ms = MeanShift(bandwidth = bandwidth,bin_seeding=True)
        ms.fit(data.cpu().data.numpy())
        for val ,label in zip(range(self.ent_tot),ms.labels_):
            transfer_dict[val] = label
        for i in range(len(np.unique(ms.labels_))):
            e_mean_dict[i] = []
        for key, category in transfer_dict.items():
            e_mean_dict[category].append(key)
        for category, list in e_mean_dict.items():
            e_mean = matrix_transfer(torch.LongTensor(list))
            e_mean = torch.mean(e_mean, dim=0)
            e_mean = F.normalize(e_mean, p=2, dim=-1)
            e_mean_dict[category] = e_mean
        return transfer_dict, e_mean_dict  # 返回实体聚类每一类均值

    def _transfer(self, e, batch_e, r_transfer,transfer_dict, e_mean_dict):
        mean_transfer = torch.zeros(len(batch_e),self.dim_e)
        i = 0
        for ent in batch_e:
            mean_transfer[i] = e_mean_dict[transfer_dict[ent.item()]]
            i +=1
            if i == len(batch_e):
                break
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1, r_transfer.shape[0], e.shape[-1])
            mean_transfer = mean_transfer.view(-1, r_transfer.shape[0], mean_transfer.shape[-1])
            r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
            e = F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * mean_transfer, -1, True) * r_transfer,
                p=2,
                dim=-1
            )
            return e.view(-1, e.shape[-1])
        else:
            return F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * mean_transfer, -1, True) * r_transfer,
                p=2,
                dim=-1
            )

    def hinge_loss(self, p_score, n_score):
        if self.margin_flag == False:
            return torch.sum(torch.max(p_score - n_score, self.zero_const))
        else:
            return torch.sum(torch.max(p_score - n_score, - self.margin) +self.margin)

    def _get_positive_score(self, score):
        positive_score = score[:self.batch_size]
        positive_score = positive_score.view(-1, self.batch_size).permute(1, 0)  # Tensor维度换位，但是不改变吗tensor的值,但是这有点不懂
        return positive_score

    def _get_negative_score(self, score):
        negative_score = score[self.batch_size:]
        negative_score = negative_score.view(-1, self.batch_size).permute(1, 0)
        return negative_score

    #定义损失函数，反向传播计算损失函数
    def forward(self, data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
#       batch_y = data['batch_y']  #不知道有什么用
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
#       h_transfer = self.ent_transfer(batch_h)
#       t_transfer = self.ent_transfer(batch_t)
        r_transfer = self.rel_transfer(batch_r)
        transfer_dict, e_mean_dict = self.meanshift(self.ent_transfer)  # 每更新一次就计算一次实体投影的平均值
        h = self._transfer(h, batch_h, r_transfer,transfer_dict, e_mean_dict)
        t = self._transfer(t, batch_t, r_transfer,transfer_dict, e_mean_dict)
        score = self._calc(h, t, r)
        #计算正负得分
        p_score = self._get_positive_score(score)
        n_score = self._get_negative_score(score)
        loss_res = self.hinge_loss(p_score, n_score)
        if self.regul_rate != 0:
            loss_res = (1-self.kl_rate)*loss_res+self.kl_rate*self.kldiv(data)+self.regul_rate * self.regularization(data)
        return loss_res

    def kldiv(self,data):
        batch_h = data['batch_h']
        batch_t = data['batch_t']

        entity_probability_matrix = torch.zeros(self.ent_tot,self.ent_tot)
        entity_transfer_probability_matrix = torch.zeros(self.ent_tot,self.ent_tot)

        #得到Tbatch所有实体集合
        batch_e = []
        for entity in torch.cat([batch_h,batch_t],dim=0):
            if entity not in batch_e:
                batch_e.append(entity)

        #得到实体e的概率距离矩阵
        for i in batch_e:
            for j in batch_e:
                p_ij_on = torch.exp(-torch.norm(self.ent_embeddings([i])-self.ent_embeddings([j]),p=2,dim=0))
                p_ij_under = torch.Tensor([0])
                p_ji_under = torch.Tensor([0])
                for k in batch_e:
                    if torch.equal(k,i):
                        p_ij_under += torch.exp(-torch.norm(self.ent_embeddings([i]) - self.ent_embeddings([k]), p=2, dim=0))
                    if torch.equal(k,j):
                        p_ij_under += torch.exp(-torch.norm(self.ent_embeddings([i]) - self.ent_embeddings([k]), p=2, dim=0))
                p_ij = torch.div(p_ij_on,2*p_ij_under)+torch.div(p_ij_on,2*p_ji_under)
                entity_probability_matrix[i][j] = p_ij
                entity_probability_matrix[j][i] = p_ij

        #得到实体映射e_p的概率距离矩阵
        for i in batch_e:
            for j in batch_e:
                q_ij_on = torch.exp(-torch.norm(self.ent_transfer([i])-self.ent_transfer([j]),p=2,dim=0))
                q_ij_under = torch.Tensor([0])
                for k in batch_e[0:-1]:
                    for l in batch_e[np.argwhere(k.cpu.numpy())+1:]:
                        q_ij_under += torch.exp(-torch.norm(self.ent_embeddings([k]) - self.ent_embeddings([l]), p=2, dim=0))
                q_ij = torch.div(q_ij_on,q_ij_under)
                entity_transfer_probability_matrix[i][j] = q_ij
                entity_transfer_probability_matrix[j][i] = q_ij

        #kl散度损失
        loss_kl = 0
        for i in self.ent_tot:
            for j in self.ent_tot:
                if (entity_transfer_probability_matrix[i][j].item()!=0)&(entity_probability_matrix[i][j].item()!=0):
                    loss_kl += entity_probability_matrix[i][j]*(torch.log(entity_probability_matrix[i][j])-torch.log(entity_transfer_probability_matrix[i][j]))
        return loss_kl

    def regularization(self, data): #正则项
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_transfer = self.ent_transfer(batch_h)
        t_transfer = self.ent_transfer(batch_t)
        r_transfer = self.rel_transfer(batch_r)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2) +
                 torch.mean(h_transfer ** 2) +
                 torch.mean(t_transfer ** 2) +
                 torch.mean(r_transfer ** 2)) / 6
        return regul

    def predict(self, data):#预测的时候需要将Tensor转换成numpy来进行,Tester.py需要
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        r_transfer = self.rel_transfer(batch_r)
        h = self._transfer(h, batch_h, r_transfer)
        t = self._transfer(t, batch_t, r_transfer)
        score = self._calc(h, t, r)
        return score.cpu().data.numpy()
