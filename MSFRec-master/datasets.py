

import torch
import random
from torch.utils.data import Dataset


#REC 括号内是继承的类
class FMLPRecDataset(Dataset):
    #REC 下划线__开头的属性或者函数是私有的，不应该被外部直接访问
    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = []
        self.max_len = args.max_seq_length
        #self.num = 0
        if data_type=='train':

            for seq in user_seq:
                """
                #REC 去掉了长度为1的预测序列
                input_ids = seq[:-2]
                for i in range(1,len(input_ids )):
                    x = input_ids[:i + 1]
                    if len(x)>50:
                        x = x[:-50]
                    self.user_seq.append(x)
                    self.num = self.num + 1
                """
                #REC 从序列的最右边开始 往前截断max_len + 2个项目 最后两个项目不要
                input_ids = seq[-(self.max_len + 2):-2]  # keeping same as train set
                #REC [1, 2, 3, 4, 5] 最后变成 [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4], [1, 2, 3, 4, 5]]
                #REC 创建子序列（用于自回归预测任务）
                for i in range(len(input_ids)):
                    self.user_seq.append(input_ids[:i + 1])


        elif data_type=='valid':
            for sequence in user_seq:
                #REC 取序列但不包含最后一个项目的 用于评估
                self.user_seq.append(sequence[:-1])
        else:
            #REC 直接使用完整序列 用于测试
            self.user_seq = user_seq

        #REC 存储负样本 数据类型 最大序列长度
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        #self.max_len = args.max_seq_length

    #REC __len__()返回数据集的大小
    def __len__(self):
        return len(self.user_seq)

    # REC __getitem__()根据索引获取数据集中的样本
    def __getitem__(self, index):
        items = self.user_seq[index]
        #REC 除了最后一个项目的序列
        input_ids = items[:-1]
        #REC 最后一个项目预测目标
        answer = items[-1]

        seq_set = set(items)
        #REC 生成预测的负样本
        neg_answer = neg_sample(seq_set, self.args.item_size)

        #REC 计算填充长度
        pad_len = self.max_len - len(input_ids)
        #REC pad_len个0组成的列表，并将其与原始的input_ids列表拼接
        input_ids = [0] * pad_len + input_ids
        input_ids = input_ids[-self.max_len:]
        #REC 确保序列的长度正好为max_len
        assert len(input_ids) == self.max_len
        # Associated Attribute Prediction
        # Masked Attribute Prediction

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]
            #REC 元组
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),#REC 填充后的用户交互序列
                #torch.tensor(attribute, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(index, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                #torch.tensor(attribute, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(neg_answer, dtype=torch.long),
            )

        return cur_tensors #REC 返回数据集中一条数据： 用户id 交互序列 预测值(正样本) 负的预测值 用于评估/测试的负样本

#REC 随机抽取一个负样本
def neg_sample(item_set, item_size):  # 前闭后闭
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item