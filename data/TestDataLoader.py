# _*_ coding:utf-8 _*_
# __author: zhangxin
import codecs
import random
import copy


class TestDataSampler(object):

    def __init__(self, data_total, data_sampler):
        self.data_total = data_total
        self.data_sampler = data_sampler
        self.total = 0

    def __iter__(self):
        return self

    def __next__(self):
        self.total += 1
        if self.total > self.data_total:
            raise StopIteration()
        return self.data_sampler(self.total)

    def __len__(self):
        return self.data_total


#导入数据,只选取一个负例
class TestDataLoader(object):

    def __init__(self,in_path,test_file,entity_set,train_triple,Filt_flag = True,sample_mode = "link"):

        self.test_file = in_path + test_file
        self.entity_set = entity_set
        self.test_triple = []
        self.Filt_flag = Filt_flag
        self.train_triple = train_triple #训练集中的正例
        self.sample_mode = sample_mode

       #加载测试三元组
        with codecs.open(self.test_file,"r") as t_f:
            lines = t_f.readlines()
            for line in lines:
                triple = line.strip().split()
                if len(triple) != 3:
                    continue
                h_ = triple[0]
                t_ = triple[1]
                r_ = triple[2]

                self.test_triple.append([h_, t_, r_])


    #得到Filt的三元组(过滤掉所有出现在训练集的三元组）
    def get_filted_triple(self,entity_set,total):

        head_triple_dict = {"batch_h":[],"batch_t":[],"batch_r":[],"mode":"head_batch"}
        tail_triple_dict = {"batch_h":[],"batch_t":[],"batch_r":[],"mode":"tail_batch"}

        triple = self.test_triple[total-1]
        for entity in entity_set:
            corrupted_head = [entity, triple[1], triple[2]]
            if corrupted_head not in self.train_triple:
                head_triple_dict["batch_h"].append(corrupted_head[0])
                head_triple_dict["batch_t"].append(corrupted_head[1])
                head_triple_dict["batch_r"].append(corrupted_head[2])

            corrupted_tail = [triple[0], entity, triple[2]]
            if corrupted_head not in self.train_triple:
                tail_triple_dict["batch_h"].append(corrupted_tail[0])
                tail_triple_dict["batch_t"].append(corrupted_tail[1])
                tail_triple_dict["batch_r"].append(corrupted_tail[2])
        corrupt_triple_list = [head_triple_dict,tail_triple_dict]
        return (triple,corrupt_triple_list)

    #得到Raw的三元组
    def get_raw_triple(self, entity_set,total):

        head_triple_dict = {"batch_h": [], "batch_t": [], "batch_r": [], "mode": "head_batch"}
        tail_triple_dict = {"batch_h": [], "batch_t": [], "batch_r": [], "mode": "tail_batch"}

        triple = self.test_triple[total-1]
        for entity in entity_set:
            corrupted_head = [entity, triple[1], triple[2]]
            head_triple_dict["batch_h"].append(corrupted_head[0])
            head_triple_dict["batch_t"].append(corrupted_head[1])
            head_triple_dict["batch_r"].append(corrupted_head[2])

            corrupted_tail = [triple[0], entity, triple[2]]
            tail_triple_dict["batch_h"].append(corrupted_tail[0])
            tail_triple_dict["batch_t"].append(corrupted_tail[1])
            tail_triple_dict["batch_r"].append(corrupted_tail[2])
        corrupt_triple_list = [head_triple_dict,tail_triple_dict]

        return (triple,corrupt_triple_list)

    def sample_lp(self,total):
        if self.Filt_flag !=False:
            return self.get_filted_triple(self.entity_set,total)
        else:
            return self.get_raw_triple(self.entity_set,total)

    def sample_tc(self,total):

        test_triple_dict = {"batch_h": [], "batch_t": [], "batch_r": [], "batch_y": []}

        triple = self.test_triple[total-1]
        corrupted_triple = copy.deepcopy(triple)
        if random.random() > 0.5:
            # 替换head
            rand_head = triple[0]  #可改进，因为没有过滤掉所有正例
            while (rand_head == triple[0]):#避免triple = corrupted_triple
                rand_head = random.sample(list(self.entity_set), 1)[0]  # [0]变成元素值
            corrupted_triple[0] = rand_head

        else:
            # 替换tail
            rand_tail = triple[1]
            while rand_tail == triple[1]:
                rand_tail = random.sample(list(self.entity_set), 1)[0]
            corrupted_triple[1] = rand_tail

        test_triple_dict["batch_h"].insert(0,triple[0])
        test_triple_dict["batch_t"].insert(0,triple[1])
        test_triple_dict["batch_r"].insert(0,triple[2])
        test_triple_dict["batch_y"].insert(0,1)

        test_triple_dict["batch_h"].append(corrupted_triple[0])
        test_triple_dict["batch_t"].append(corrupted_triple[1])
        test_triple_dict["batch_r"].append(corrupted_triple[2])
        test_triple_dict["batch_y"].append(0)

        return test_triple_dict

    def get_triple_tot(self):
        return len(self.test_triple)

    def __iter__(self):
        if self.sample_mode == "link":
            return TestDataSampler(self.get_triple_tot(), self.sample_lp)
        else:
            return TestDataSampler(self.get_triple_tot(), self.sample_tc)

    def set_sample_mode(self, sample_mode):
        self.sample_mode = sample_mode






