

import tqdm
import torch
import numpy as np

from torch.optim import Adam
from utils import recall_at_k, ndcg_k, get_metric

class Trainer:
    def __init__(self, model, train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):

        self.args = args
        self.cuda_condition = torch.cuda.is_available() and not self.args.no_cuda
        self.device = torch.device("cuda" if self.cuda_condition else "cpu")

        self.model = model
        #REC cuda可用的话 将模型移动到GPU
        if self.cuda_condition:
            self.model.cuda()  #REC ？？？？

        # Setting the train and test data loader
        self.train_dataloader = train_dataloader
        self.eval_dataloader = eval_dataloader
        self.test_dataloader = test_dataloader
        #REC 使用args中的超参数设置Adam优化器????
        # self.data_name = self.args.data_name
        betas = (self.args.adam_beta1, self.args.adam_beta2)
        self.optim = Adam(self.model.parameters(), lr=self.args.lr, betas=betas, weight_decay=self.args.weight_decay)
        #REC 打印模型中的总参数数量
        print("Total Parameters:", sum([p.nelement() for p in self.model.parameters()]))
    #REC
    def train(self, epoch):
        self.iteration(epoch, self.train_dataloader)
    #REC
    def valid(self, epoch, full_sort=False):
        if full_sort:
            self.args.train_matrix = self.args.valid_rating_matrix
        return self.iteration(epoch, self.eval_dataloader, full_sort, train=False)
    #REC
    def test(self, epoch, full_sort=False):
        if full_sort:
            self.args.train_matrix = self.args.test_rating_matrix
        return self.iteration(epoch, self.test_dataloader, full_sort, train=False)

    def iteration(self, epoch, dataloader, full_sort=False, train=True):
        raise NotImplementedError
    #REC test使用的 pred_list 是一个列表，其中包含了每个样本中真实答案的排名信息
    def get_sample_scores(self, epoch, pred_list):
        #REC 获取原序列中每个元素的排名  按照原顺序
        #REC pred_list维度为[num_users,1] 返回正确答案的排名
        pred_list = (-pred_list).argsort().argsort()[:, 0]
        HIT_1, NDCG_1, MRR = get_metric(pred_list, 1)
        HIT_5, NDCG_5, MRR = get_metric(pred_list, 5)
        HIT_10, NDCG_10, MRR = get_metric(pred_list, 10)
        post_fix = {
            "Epoch": epoch,
            "HIT@1": '{:.4f}'.format(HIT_1), "NDCG@1": '{:.4f}'.format(NDCG_1),
            "HIT@5": '{:.4f}'.format(HIT_5), "NDCG@5": '{:.4f}'.format(NDCG_5),
            "HIT@10": '{:.4f}'.format(HIT_10), "NDCG@10": '{:.4f}'.format(NDCG_10),
            "MRR": '{:.4f}'.format(MRR),
        }
        print(post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)

    def get_full_sort_score(self, epoch, answers, pred_list):
        #REC answer [num_vaild_user,1] pred_list [num_vaild_user,20]
        recall, ndcg = [], []
        for k in [5, 10, 15, 20]:
            recall.append(recall_at_k(answers, pred_list, k))
            ndcg.append(ndcg_k(answers, pred_list, k))
        post_fix = {
            "Epoch": epoch,
            "HIT@5": '{:.4f}'.format(recall[0]), "NDCG@5": '{:.4f}'.format(ndcg[0]),
            "HIT@10": '{:.4f}'.format(recall[1]), "NDCG@10": '{:.4f}'.format(ndcg[1]),
            "HIT@20": '{:.4f}'.format(recall[3]), "NDCG@20": '{:.4f}'.format(ndcg[3])
        }
        print("get_full_sort_score ",post_fix)
        with open(self.args.log_file, 'a') as f:
            f.write(str(post_fix) + '\n')
        return [recall[0], ndcg[0], recall[1], ndcg[1], recall[3], ndcg[3]], str(post_fix)
    #REC 将模型的参数保存到文件中
    def save(self, file_name):
        torch.save(self.model.cpu().state_dict(), file_name)
        self.model.to(self.device)
    #REC 从文件中加载预训练的模型参数
    def load(self, file_name):
        original_state_dict = self.model.state_dict()
        print(original_state_dict.keys())
        #REC 加载预训练模型的参数
        new_dict = torch.load(file_name)
        print(new_dict.keys())
        #REC 更新当前模型的参数
        for key in new_dict:
            original_state_dict[key]=new_dict[key]
        self.model.load_state_dict(original_state_dict)
    #REC 损失函数
    def cross_entropy(self, seq_out, pos_ids, neg_ids):
        #REC seq_out的维度[batch seq_len hidden_size]
        pos_emb = self.model.item_embeddings(pos_ids)
        neg_emb = self.model.item_embeddings(neg_ids)

        # [batch hidden_size]
        #pos = pos_emb.view(-1, pos_emb.size(2))
        #neg = neg_emb.view(-1, neg_emb.size(2))
        #REC 用于预测下一项的序列嵌入
        seq_emb = seq_out[:, -1, :] # [batch_size,1, hidden_size]
        #REC 序列嵌入和正样本嵌入的按元素点积 沿嵌入维度求和
        pos_logits = torch.sum(pos_emb * seq_emb, -1) # [batch_size]
        #REC 序列嵌入和负样本嵌入的按元素点积 沿嵌入维度求和
        neg_logits = torch.sum(neg_emb * seq_emb, -1)
        #istarget = (pos_ids > 0).view(pos_ids.size(0) * self.model.args.max_seq_length).float() # [batch*seq_len]
        loss = torch.mean(
            - torch.log(torch.sigmoid(pos_logits) + 1e-24) -
            torch.log(1 - torch.sigmoid(neg_logits) + 1e-24)
        )# / torch.sum(istarget)

        return loss
    #REC  test使用的 负样本和序列嵌入的得分 seq_out[batch_size,1 最后一个,hidden_size] test_neg_sample[atch_size,100]
    def predict_sample(self, seq_out, test_neg_sample):
        # [batch 100 hidden_size]
        test_item_emb = self.model.item_embeddings(test_neg_sample)
        # [batch 100 hidden_size] * [batch_size, hidden_size,1] = [batch 100 1]
        test_logits = torch.bmm(test_item_emb, seq_out.unsqueeze(-1)).squeeze(-1)  # [B 100]
        return test_logits
    #REC 所有项目和序列嵌入的得分
    def predict_full(self, seq_out):
        # [item_num hidden_size]
        test_item_emb = self.model.item_embeddings.weight
        # [batch hidden_size ]
        #REC seq_out [batch，hidden_size] test_item_emb [item_num hidden_size]
        #REC rating_pred [batch，item_num ]
        rating_pred = torch.matmul(seq_out, test_item_emb.transpose(0, 1))
        return rating_pred

class FMLPRecTrainer(Trainer):

    def __init__(self, model,
                 train_dataloader,
                 eval_dataloader,
                 test_dataloader, args):
        super(FMLPRecTrainer, self).__init__(
            model,
            train_dataloader,
            eval_dataloader,
            test_dataloader, args
        )

    def iteration(self, epoch, dataloader, full_sort=False, train=True):

        str_code = "train" if train else "test"

        # Setting the tqdm progress bar
        #REC 创建一个进度条，用于在训练或测试推荐系统时显示迭代进度
        #REC 将数据加载器 dataloader 中的每个批次与其索引一起返回
        #REC str_code代表训练or测试 epoch 是当前的训练周期
        #REC 数据加载器中的批次总数
        rec_data_iter = tqdm.tqdm(enumerate(dataloader),#REC 获得batch的索引和batch的数据
                                  desc="Recommendation EP_%s:%d" % (str_code, epoch),
                                  total=len(dataloader),
                                  bar_format="{l_bar}{r_bar}")
        if train:
            self.model.train()
            rec_loss = 0.0
            #REC 遍历数据迭代器 i 是批次索引，batch 是批次数据。
            for i, batch in rec_data_iter:
                # 0. batch_data will be sent into the device(GPU or CPU)
                #REC 将批次数据发送到指定的设备
                batch = tuple(t.to(self.device) for t in batch)
                _, input_ids, answer, neg_answer = batch
                # Binary cross_entropy
                #REC [batch_size,seq_len,enbed_len]
                sequence_output = self.model(input_ids)

                loss = self.cross_entropy(sequence_output, answer, neg_answer)
                #REC 清零之前的梯度，为新的批次准备。
                self.optim.zero_grad()
                #REC 进行反向传播，计算梯度
                loss.backward()
                #REC 根据计算的梯度更新模型参数
                self.optim.step()
                #REC 计算当前batch的累计损失
                rec_loss += loss.item()


            post_fix = {
                "epoch": epoch,
                "rec_loss": '{:.4f}'.format(rec_loss / len(rec_data_iter)),#REC 每个epoch的平均损失
            }
            #REC 每个epoch打印一次日志
            if (epoch + 1) % self.args.log_freq == 0:
                print(str(post_fix))

            with open(self.args.log_file, 'a') as f:
                f.write(str(post_fix) + '\n')

        else:
            self.model.eval()

            pred_list = None

            if full_sort:
                answer_list = None
                #REC i是批次索引，batch是批次数据。
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, answers, _, neg_answer = batch
                    #REC recommend_output维度为[batch seq_len hidden_size]
                    recommend_output = self.model(input_ids)
                    # REC recommend_output维度为[batch，1， hidden_size]
                    recommend_output = recommend_output[:, -1, :]# 推荐的结果
                    # REC rating_pred[batch，num_item]
                    rating_pred = self.predict_full(recommend_output)
                    #REC 将PyTorch张量（tensor）转换为NumPy数组，并确保结果在CPU上
                    rating_pred = rating_pred.cpu().data.numpy().copy()
                    batch_user_index = user_ids.cpu().numpy()
                    #REC 将用户已经交互过的物品的分数置为0
                    rating_pred[self.args.train_matrix[batch_user_index].toarray() > 0] = 0
                    # reference: https://stackoverflow.com/a/23734295, https://stackoverflow.com/a/20104162
                    # argpartition 时间复杂度O(n)  argsort O(nlogn) 只会做
                    # 加负号"-"表示取大的值
                    #REC  快速找到前20个的索引 Top-K 的元素
                    ind = np.argpartition(rating_pred, -20)[:, -20:]#[batch，num_item]
                    # 根据返回的下标 从对应维度分别取对应的值 得到每行topk的子表
                    arr_ind = rating_pred[np.arange(len(rating_pred))[:, None], ind]
                    # 对子表进行排序 得到从大到小的顺序
                    arr_ind_argsort = np.argsort(arr_ind)[np.arange(len(rating_pred)), ::-1]
                    # REC 分数从高到低的前20个物品索引[batch_size,20]
                    batch_pred_list = ind[np.arange(len(rating_pred))[:, None], arr_ind_argsort]

                    if i == 0:
                        pred_list = batch_pred_list
                        answer_list = answers.cpu().data.numpy()
                    else:
                        pred_list = np.append(pred_list, batch_pred_list, axis=0)
                        answer_list = np.append(answer_list, answers.cpu().data.numpy(), axis=0)
                return self.get_full_sort_score(epoch, answer_list, pred_list)

            else:
                for i, batch in rec_data_iter:
                    # 0. batch_data will be sent into the device(GPU or cpu)
                    batch = tuple(t.to(self.device) for t in batch)
                    user_ids, input_ids, answers, _, sample_negs = batch
                    #REC [batch seq_len hidden_size] [256,50,64]
                    recommend_output = self.model(input_ids)
                    #REC [(batch_size, num_answers + num_negatives=100]
                    test_neg_items = torch.cat((answers.unsqueeze(-1), sample_negs), -1)
                    #REC [batch hidden_size]
                    recommend_output = recommend_output[:, -1, :]
                    #REC [batch 100 ] 100项目【0:真实项目 1-99 负抽样的项目】和预测项目的点积
                    test_logits = self.predict_sample(recommend_output, test_neg_items)
                    test_logits = test_logits.cpu().detach().numpy().copy()
                    if i == 0:
                        pred_list = test_logits
                    else:
                        pred_list = np.append(pred_list, test_logits, axis=0)
                #pred_list [22363,100]
                return self.get_sample_scores(epoch, pred_list)
