# _*_ coding:utf-8 _*_
# __author: zhangxin

from data.TrainDataLoader import TrainDataLoader
from data.TestDataLoader import TestDataLoader
from model.Model import PTransD
#from model.BasicModule import BaseModule
from config.Trainer import Trainer
from config.Tester import Tester


#加载训练数据
train_dataloader = TrainDataLoader(in_path="./benchmarks/WN18/", batch_size=100, sampling_mode="unif")

#加载测试数据
test_dataloader = TestDataLoader(in_path="./benchmarks/WN18/",test_file = "valid2id.txt",entity_set=set(train_dataloader.entity2id.values()),train_triple = train_dataloader.triple_list,sample_mode="link")
#test_dataloader = TestDataLoader(in_path="./benchmarks/FB15K237/",entity_set=train_dataloader.entity_set,train_triple = train_dataloader.triple_list,sample_mode="classification")

# 实例化模型
ptransd = PTransD(ent_tot = train_dataloader.get_ent_tot(),
                  rel_tot = train_dataloader.get_rel_tot(),
                  dim_e=20,
                  dim_r=20,
                  p_norm=1,
                  norm_flag=True,
                  margin=1.0,
                  k = 5,
                  epsilon=1000,
                  batch_size = train_dataloader.batch_size,
                  regul_rate = 0.0,
                  kl_rate = 0.3)

# 训练模型
trainer = Trainer(model = ptransd, data_loader = train_dataloader, train_times = 1000, alpha = 1.0, use_gpu = False)
trainer.run()
ptransd.save_checkpoint('./checkpoints/ptransd.ckpt')   #保存一下这个模型

#测试模型
ptransd.load_checkpoint('./checkpoints/ptransd.ckpt')
tester = Tester(batch_size =train_dataloader.batch_size, model = ptransd, data_loader = test_dataloader, use_gpu = False)
tester.run_link_prediction()
#tester.run_triple_classification()




























