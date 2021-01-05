import codecs
import random
import math
import numpy as np
import copy
import time


class TrainDataSampler(object):

    def __init__(self, nbatches, datasampler):
        self.nbatches = nbatches
        self.datasampler = datasampler
        self.batch = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.batch += 1
        if self.batch > self.nbatches:
            raise StopIteration()
        return self.datasampler()

    def __len__(self):
        return self.nbatches

#导入数据,只选取一个负例
class TrainDataLoader(object):

    def __init__(self,batch_size,in_path,sampling_mode=None):

        self.batch_size = batch_size
        self.entity_set = set()
        self.relation_set = set()
        self.triple_list = []
        self.sampling_mode = sampling_mode
        self.file1 = in_path + "train2id.txt"
        self.file2 = in_path + "entity2id.txt"
        self.file3 = in_path + "relation2id.txt"
        self.entity2id = {}
        self.relation2id = {}
        self.batch = 0

        with open(self.file2, 'r') as f1, open(self.file3, 'r') as f2:
            lines1 = f1.readlines()
            lines2 = f2.readlines()
            for line in lines1:
                line = line.strip().split('\t')
                if len(line) != 2:
                 continue
                self.entity2id[line[0]] = line[1]

            for line in lines2:
                line = line.strip().split('\t')
                if len(line) != 2:
                    continue
                self.relation2id[line[0]] = line[1]


        self.head_list = {}  # 所有的头实体集合字典,每个元素是 实体：[以该实体为头实体的尾实体]
        self.tail_list = {}

        with codecs.open(self.file1, 'r') as f:
            content = f.readlines()
            for line in content:
                triple = line.strip().split()
                if len(triple) != 3:
                    continue
                h_ = triple[0]
                t_ = triple[1]
                r_ = triple[2]

                if h_ not in self.head_list:
                    self.head_list[h_] = [t_]
                else:
                    self.head_list[h_].append(t_)
                if t_ not in self.tail_list:
                    self.tail_list[t_] = [h_]
                else:
                    self.tail_list[t_].append(h_)

#                self.entity_set.add(h_)
#                self.entity_set.add(t_)
#                self.relation_set.add(r_)
                self.triple_list.append([h_,t_,r_])

#            self.entTotal = len(self.entity_set) #训练集实体数目
#            self.relTotal = len(self.relation_set) #训练集关系数目
            self.tripleTotal = len(self.triple_list) #训练集数目

        total_tail_per_head = 0
        total_head_per_tail = 0

        for h in self.head_list:
            total_tail_per_head += len(self.head_list[h])
        for t in self.tail_list:
            total_head_per_tail += len(self.tail_list[t])
        self.tph = 0  # 每个头实体平均几个尾实体
        self.hpt = 0  # 每个尾实体平均几个头实体
        self.tph = total_tail_per_head / len(self.head_list)
        self.hpt = total_head_per_tail / len(self.tail_list)

        self.nbatches = self.tripleTotal // self.batch_size #结果向下取整

    def sample(self):
        # Sbatch:list
        Sbatch = random.sample(self.triple_list, self.batch_size) #有点问题的，因为sample随机取batch_size一迭代会重复，先放着,一定要该，我觉得不对
        Tbatch = {'batch_h': [], 'batch_t': [], 'batch_r': [], 'batch_y': []}

        for triple in Sbatch:
            if self.sampling_mode == "unif":
                Tbatch = self.sample_unif(triple,Tbatch)
            else:
                Tbatch = self.sample_bern(triple, Tbatch)

        return Tbatch

    def sample_unif(self,triple,Tbatch):

        corrupted_triple = copy.deepcopy(triple)
        if random.random() > 0.5:
            # 替换head
            rand_head = triple[0]

            while rand_head == triple[0]:#避免triple = corrupted_triple
                rand_head = random.sample(list(self.entity2id.values()), 1)[0]  # [0]变成元素值
            corrupted_triple[0] = rand_head

        else:
            # 替换tail
            rand_tail = triple[1]
            while rand_tail == triple[1]:
                rand_tail = random.sample(list(self.entity2id.values()), 1)[0]
            corrupted_triple[1] = rand_tail

        Tbatch['batch_h'].insert(0,triple[0])
        Tbatch['batch_t'].insert(0,triple[1])
        Tbatch['batch_r'].insert(0,triple[2])
        Tbatch['batch_y'].insert(0,1)

        Tbatch['batch_h'].append(corrupted_triple[0])
        Tbatch['batch_t'].append(corrupted_triple[1])
        Tbatch['batch_r'].append(corrupted_triple[2])
        Tbatch['batch_y'].append(0)

        return Tbatch

    def sample_bern(self,triple,Tbatch):

        corrupted_triple = copy.deepcopy(triple)
        if random.random() < (self.tph / (self.tph + self.hpt)):
            corrupted_triple[0] = np.random.choice(list(self.entity2id.values()),1)[0]
        else:
            corrupted_triple[2] = np.random.choice(list(self.entity2id.values()),1)[0]


        Tbatch['batch_h'].append(triple[0])
        Tbatch['batch_t'].append(triple[1])
        Tbatch['batch_r'].append(triple[2])
        Tbatch['batch_y'].append(1)

        Tbatch['batch_h'].append(corrupted_triple[0])
        Tbatch['batch_t'].append(corrupted_triple[1])
        Tbatch['batch_r'].append(corrupted_triple[2])
        Tbatch['batch_y'].append(0)

        return Tbatch

    def __iter__(self):
        return TrainDataSampler(self.nbatches, self.sample)

    def get_ent_tot(self):
        return len(self.entity2id)

    def get_rel_tot(self):
        return len(self.relation2id)

    def get_triple_tot(self):
        return self.tripleTotal

if __name__ == "__main__":
    train_dataloader = TrainDataLoader(in_path="../benchmarks/FB15K237/", batch_size=2000, sampling_mode="unif")
#    for data in train_dataloader:
#        print(data)
    print(len(train_dataloader.entity2id))
    print(len(train_dataloader.entity_set))