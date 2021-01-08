# _*_ coding:utf-8 _*_
# __author: zhangxin
import torch
from torch.autograd import Variable
import numpy as np
from tqdm import tqdm


class Tester(object):

    def __init__(self,batch_size,model = None, data_loader = None, use_gpu = True):

        self.model = model
        self.data_loader = data_loader
        self.use_gpu = use_gpu
        self.batch_size = batch_size

        if self.use_gpu:
            self.model.cuda()

    def set_model(self, model):
        self.model = model

    def set_data_loader(self, data_loader):
        self.data_loader = data_loader

    def set_use_gpu(self, use_gpu):
        self.use_gpu = use_gpu
        if self.use_gpu and self.model != None:
            self.model.cuda()

    def to_var(self, x, use_gpu):
        if use_gpu:
            return Variable(torch.LongTensor(list(map(int, x))).cuda())
        else:
            return Variable(torch.LongTensor(list(map(int, x))))

    def test_one_step(self, data):
        return self.model.predict({             #不懂这里为什么需要变量
            'batch_h': self.to_var(data['batch_h'], self.use_gpu),
            'batch_t': self.to_var(data['batch_t'], self.use_gpu),
            'batch_r': self.to_var(data['batch_r'], self.use_gpu),
        })

    def run_link_prediction(self):
        hits10 = 0
        rank_sum = 0

        self.data_loader.set_sample_mode('link')
        training_range = tqdm(self.data_loader)
        for index, data in enumerate(training_range):

            triple = data[0]
            data_head = data[1][0]
            data_tail = data[1][1]

            head_score = self.test_one_step(data_head)
            order = np.argsort(head_score)

            for rank in order:
                if data_head["batch_h"][rank]== triple[0]:
                    if rank<10:
                        hits10 +=1
                    rank_sum = rank_sum+rank+1
                    break

            tail_score = self.test_one_step(data_tail)
            order = np.argsort(tail_score)

            for rank in order:
                if data_tail["batch_t"][rank] == triple[1]:
                    if rank < 10:
                        hits10 += 1
                    rank_sum = rank_sum+rank+1
                    break

        hits10 = hits10 / 2*self.data_loader.get_triple_tot()
        mean_rank = rank_sum / 2*self.data_loader.get_triple_tot()

        print(hits10,mean_rank)
        return hits10,mean_rank

    def get_best_threshlod(self, score, ans):
        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score) #得到score的升序的索引
        res = res[order]#得到2维数组(升序，先正例后负例），每个元素是[正例或负例,得分]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans) #正例的个数
        total_false = total_all - total_true #负例的个数

        res_mx = 0.0 #最大的acc值
        threshlod = None
        for index, [ans, score] in enumerate(res):
            if ans == 1:
                total_current += 1.0
            res_current = (2 * total_current + total_false - index - 1) / total_all #此时的acc值
            if res_current > res_mx:
                res_mx = res_current
                threshlod = score
        return threshlod, res_mx #acc取最大值时的score和acc

    def run_triple_classification(self, threshlod = None):
        self.data_loader.set_sample_mode('classification')
        score = []
        ans = []
        training_range = tqdm(self.data_loader)
        for index, data in enumerate(training_range):
            _score = self.test_one_step(data)
            ans = ans + [1,0]
            score = score+list(_score)

        score = np.array(score)
        ans = np.array(ans)

        if threshlod == None:
            threshlod, _ = self.get_best_threshlod(score, ans)

        res = np.concatenate([ans.reshape(-1,1), score.reshape(-1,1)], axis = -1)
        order = np.argsort(score)
        res = res[order]

        total_all = (float)(len(score))
        total_current = 0.0
        total_true = np.sum(ans)
        total_false = total_all - total_true

        for index, [ans, score] in enumerate(res):
            if score > threshlod:
                acc = (2 * total_current + total_false - index) / total_all
                break
            elif ans == 1:
                total_current += 1.0
        print(acc)
        return acc, threshlod



