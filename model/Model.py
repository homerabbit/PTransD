# _*_ coding:utf-8 _*_
# __author: zhangxin
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.BasicModule import BaseModule
from kmeans_pytorch import kmeans

class PTransD(BaseModule):

    def __init__(self,
                 ent_tot,
                 rel_tot,
                 dim_e=100,
                 dim_r=100,
                 k = 100,
                 p_norm=1,#得分用的一范
                 norm_flag=1,#实体向量需要单位化
                 margin=None,
                 epsilon=None,
                 batch_size = None,
                 regul_rate = None,
                 kl_rate = None,
                 device = "cpu"):
        super(PTransD, self).__init__()
        self.dim_e = dim_e
        self.dim_r = dim_r
        self.k = k
        self.margin = margin
        self.epsilon = epsilon
        self.batch_size = batch_size
        self.p_norm = p_norm
        self.norm_flag = norm_flag
        self.ent_tot = ent_tot
        self.rel_tot = rel_tot
        self.kl_rate = kl_rate
        self.regul_rate = regul_rate
        self.device = torch.device(device)


        #存储固定大小的词典的嵌入向量查找表
        self.ent_embeddings = nn.Embedding(self.ent_tot, self.dim_e)
        self.rel_embeddings = nn.Embedding(self.rel_tot, self.dim_r)
        self.ent_transfer = nn.Embedding(self.k, self.dim_e)
        self.rel_transfer = nn.Embedding(self.rel_tot, self.dim_r)

        self.category_dict = {}
        self.e_mean_dict = {}
        self.entity_probability_matrix = torch.zeros(size = (self.k, self.k)).to(self.device)
        self.entity_transfer_probability_matrix = torch.zeros(size = (self.k, self.k)).to(self.device)

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

        if self.regul_rate != None:
            self.regul_rate = nn.Parameter(torch.Tensor([regul_rate]))
            self.regul_rate.requires_grad = False

        if self.kl_rate != None:
            self.kl_rate = nn.Parameter(torch.Tensor([kl_rate]))
            self.kl_rate .requires_grad = False

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

    def cluster_kmeans(self, matrix, k):  # 返回{transfer标号：对应的类}
        """k-means聚类算法

        k       - 指定分簇数量
        matrix     - ndarray(m, n)，m个样本的数据集，每个样本n个属性值
        """
        data = matrix.weight.to(self.device)
        cluster_ids_x, cluster_centers = kmeans(X=data, num_clusters=self.k, distance='euclidean', device=self.device)

        for i in range(self.ent_tot):
            self.category_dict[i] = cluster_ids_x[i]

        for i in range(k):
            self.e_mean_dict[i] = cluster_centers[i]

    def _transfer(self, e, e_transfer, r_transfer,):
        if e.shape[0] != r_transfer.shape[0]:
            e = e.view(-1, r_transfer.shape[0], e.shape[-1])
            e_transfer = e_transfer.view(-1, r_transfer.shape[0], e_transfer.shape[-1])
            r_transfer = r_transfer.view(-1, r_transfer.shape[0], r_transfer.shape[-1])
            e = F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                p=2,
                dim=-1
            )
            return e.view(-1, e.shape[-1])
        else:
            return F.normalize(
                self._resize(e, -1, r_transfer.size()[-1]) + torch.sum(e * e_transfer, -1, True) * r_transfer,
                p=2,
                dim=-1
            )
    def category_transfer(self,batch):
        list = []
        for e in batch:
            list.append(self.category_dict[e.item()])
        return torch.LongTensor(list).to(self.device)

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
        self.cluster_kmeans(self.ent_embeddings, self.k)  # 每更新一次就计算一次所有实体的平均值
        h_transfer = self.ent_transfer(self.category_transfer(batch_h))
        t_transfer = self.ent_transfer(self.category_transfer(batch_t))
        r_transfer = self.rel_transfer(batch_r)
        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)
        score = self._calc(h, t, r)
        #计算正负得分
        p_score = self._get_positive_score(score)
        n_score = self._get_negative_score(score)
        loss_res = self.hinge_loss(p_score, n_score)
        if self.regul_rate != 0:
            loss_res = (1-self.kl_rate)*loss_res+self.kl_rate*self.kldiv()+self.regul_rate * self.regularization(data)
        else:
            loss_res = (1 - self.kl_rate) * loss_res + self.kl_rate * self.kldiv()
        return loss_res

    def kldiv(self):
        #构建实体向量矩阵
        matrix = torch.tensor([]).to(self.device)
        for _,mean in self.e_mean_dict.items():
            matrix = torch.cat((matrix,mean.data),0)
        matrix = matrix.view(self.k, self.dim_e)
        #构建实体投影向量矩阵
        matrix_p = self.ent_transfer(torch.LongTensor([range(0,self.k)]).to(self.device)).squeeze()

        #构建概率矩阵
        q_ij_under = self.zero_const
        for i in range(self.k-1):
            for j in range(i+1,self.k):
                q_ij_under += torch.exp(-torch.norm(matrix_p[i]-matrix_p[j],p=2,dim=0))

        for i in range(self.k):
            for j in range(self.k):
                #求映射e的概率距离矩阵qij
                q_ij_on = torch.exp(-torch.norm(matrix_p[i] - matrix_p[j], p=2, dim=0))
                q_ij = torch.div(q_ij_on, q_ij_under)
                #求e的概率距离矩阵pij
                if i==j:
                    p_ij = self.zero_const
                else:
                    p_ij_on = torch.exp(-torch.norm(matrix[i]-matrix[j],p=2,dim=0))
                    p_ij_under = self.zero_const
                    p_ji_under = self.zero_const
                    for k1 in range(self.k):
                        if k1==i:
                            break
                        else:
                            p_ij_under += torch.exp(-torch.norm(matrix[i] - matrix[k1], p=2, dim=0))
                    for k2 in range(self.k):
                        if k2==j:
                            break
                        else:
                            p_ji_under += torch.exp(-torch.norm(matrix[j] - matrix[k2], p=2, dim=0))
                    p_ij = torch.div(p_ij_on, 2*self.k * p_ij_under) + torch.div(p_ij_on, 2 *self.k* p_ji_under)

                self.entity_probability_matrix[i][j] = p_ij
                self.entity_probability_matrix[j][i] = p_ij
                self.entity_transfer_probability_matrix[i][j] = q_ij
                self.entity_transfer_probability_matrix[j][i] = q_ij

        self.entity_probability_matrix = F.normalize(self.entity_probability_matrix, 2, -1)
        self.entity_transfer_probability_matrix = F.normalize(self.entity_transfer_probability_matrix, 2, -1)

        #kl散度损失
        loss_kl = 0
        for i in range(self.k):
            for j in range(self.k):
                if (self.entity_transfer_probability_matrix[i][j].item()!=0)and(self.entity_probability_matrix[i][j].item()!=0):
                    loss_kl = loss_kl+self.entity_probability_matrix[i][j]*(torch.log(self.entity_probability_matrix[i][j])-torch.log(self.entity_transfer_probability_matrix[i][j]))
        return loss_kl

    def regularization(self, data): #正则项
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        h_transfer = self.ent_transfer(self.category_transfer(batch_h))
        t_transfer = self.ent_transfer(self.category_transfer(batch_t))
        r_transfer = self.rel_transfer(batch_r)
        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)
        regul = (torch.mean(h ** 2) +
                 torch.mean(t ** 2) +
                 torch.mean(r ** 2) +
                 torch.mean(h_transfer ** 2) +
                 torch.mean(t_transfer ** 2) +
                 torch.mean(r_transfer ** 2)) / 6
        return regul

    def predict(self, data,):#预测的时候需要将Tensor转换成numpy来进行,Tester.py需要
        batch_h = data['batch_h']
        batch_t = data['batch_t']
        batch_r = data['batch_r']
        h = self.ent_embeddings(batch_h)
        t = self.ent_embeddings(batch_t)
        r = self.rel_embeddings(batch_r)
        #加载保存的category_dict
        category_transfer = torch.load("checkpoints/trained_categorydict.pth")
        h_transfer = self.ent_transfer(category_transfer(batch_h))
        t_transfer = self.ent_transfer(category_transfer(batch_t))
        r_transfer = self.rel_transfer(batch_r)
        h = self._transfer(h, h_transfer, r_transfer)
        t = self._transfer(t, t_transfer, r_transfer)
        score = self._calc(h, t, r)
        return score.cpu().data.numpy()
