

import os
import torch
import argparse
import numpy as np

from models import FMLPRecModel
from trainers import FMLPRecTrainer
from utils import EarlyStopping, check_path, set_seed, get_local_time, get_seq_dic, get_dataloder, get_rating_matrix

def main():
    #REC 定义一个命令行接口 下面是定义的命令行参数
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/", type=str) #REC指定数据集位置
    parser.add_argument("--output_dir", default="output/", type=str)
    parser.add_argument("--data_name", default="Beauty", type=str) #REC指定数据集
    parser.add_argument("--do_eval", action="store_true") #REC是否验证
    parser.add_argument("--load_model", default="MSFRec-Beauty-4eval", type=str) #REC指定模型
    #REC 带--是可选参数 没有说明该参数 default是默认值 type是参数的类型
    #REC action是开关类的参数 如果说明了该参数 对应的参数值被设置为true 反之设置为false

    # model args
    parser.add_argument("--model_name", default="MSFRec", type=str)
    parser.add_argument("--hidden_size", default=64, type=int, help="hidden size of model")
    #REC 项目嵌入向量的长度
    parser.add_argument("--num_hidden_layers", default=2, type=int, help="number of filter-enhanced blocks")
    parser.add_argument("--num_attention_heads", default=2, type=int)
    parser.add_argument("--hidden_act", default="gelu", type=str) # gelu relu
    parser.add_argument("--attention_probs_dropout_prob", default=0.5, type=float)
    parser.add_argument("--hidden_dropout_prob", default=0.5, type=float)
    parser.add_argument("--initializer_range", default=0.02, type=float)
    parser.add_argument("--stft_type", default=4, type=int, help="[1,2,3,4]")
    #REC？？？？？初始化范围 由项目最大ID确定？？？？
    parser.add_argument("--max_seq_length", default=50, type=int)
    parser.add_argument("--no_filters", action="store_true", help="if no filters, filter layers transform to self-attention")

    # train args
    parser.add_argument("--lr", default=0.001, type=float, help="learning rate of adam")
    parser.add_argument("--batch_size", default=256, type=int, help="number of batch_size") #REC 默认模型每次迭代同时处理256个训练样本
    parser.add_argument("--epochs", default=200, type=int, help="number of epochs")#REC 默认完整遍历训练数据集200次
    parser.add_argument("--no_cuda", action="store_true")
    parser.add_argument("--log_freq", default=1, type=int, help="per epoch print res")
    parser.add_argument("--full_sort", action="store_true")
    #REC 全排序 通常指的是对所有可能的候选项目进行排序，以生成一个完整的推荐列表
    parser.add_argument("--patience", default=10, type=int, help="how long to wait after last time validation loss improved")#REC 早停批次

    parser.add_argument("--seed", default=42, type=int)#42
    parser.add_argument("--weight_decay", default=0.0, type=float, help="weight_decay of adam")#REC 模型权重的L2范数（即权重的平方和）。  n
    parser.add_argument("--adam_beta1", default=0.9, type=float, help="adam first beta value")
    parser.add_argument("--adam_beta2", default=0.999, type=float, help="adam second beta value")
    parser.add_argument("--gpu_id", default="0", type=str, help="gpu_id")
    parser.add_argument("--variance", default=5, type=float)

    args = parser.parse_args()

    set_seed(args.seed)
    check_path(args.output_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.cuda_condition = torch.cuda.is_available() and not args.no_cuda

    #REC 预处理数据集 获得user_seq和sample_seq
    #REC  seq_dic = {'user_seq':user_seq, 'num_users':num_users, 'sample_seq':sample_seq}
    seq_dic, max_item = get_seq_dic(args)
    #REC 数据集中最大项目索引
    args.item_size = max_item + 1

    # save model args
    cur_time = get_local_time()
    if args.no_filters:
        args.model_name = "SASRec"
    args_str = f'{args.model_name}-{args.data_name}-{cur_time}'
    #REC 日志文件路径 拼接了模型名称、数据集名称 以及 运行时间
    args.log_file = os.path.join(args.output_dir, args_str + '.txt')
    #REC 控制台打印args 打开日志文件把args写进去
    print(str(args))
    with open(args.log_file, 'a') as f:
        f.write(str(args) + '\n')

    # save model
    args.checkpoint_path = os.path.join(args.output_dir, args_str + '.pt')
    #REC 划分数据集
    #REC  __getitem__ 函数返回index input_ids answer neg_answer test_samples
    train_dataloader, eval_dataloader, test_dataloader = get_dataloder(args,seq_dic)

    #REC 初始化模型`
    model = FMLPRecModel(args=args)
    trainer = FMLPRecTrainer(model, train_dataloader, eval_dataloader,
                              test_dataloader, args)
    #args.do_eval = True

    if args.full_sort:
        #REC 稀疏矩阵 行是所有用户  列是所有项目 交互序列除最后俩外有交互就是1
        args.valid_rating_matrix, args.test_rating_matrix = get_rating_matrix(args.data_name, seq_dic, max_item)
    #REC 是都是仅仅测试某个模型
    if args.do_eval:
        if args.load_model is None:
            print(f"No model input!")
            exit(0)
        else:
            args.checkpoint_path = os.path.join(args.output_dir, args.load_model + '.pt')
            trainer.load(args.checkpoint_path)
            print(f"Load model from {args.checkpoint_path} for test!")
            scores, result_info = trainer.test(0, full_sort=args.full_sort)
    #REC 从头训练、验证和测试
    else:
        early_stopping = EarlyStopping(args.checkpoint_path, patience=args.patience, verbose=True)
        for epoch in range(args.epochs):
            trainer.train(epoch)
            #REC 返回的是[HIT_1, NDCG_1, HIT_5, NDCG_5, HIT_10, NDCG_10, MRR], str(post_fix)
            #REC 在验证集上验证
            scores, _ = trainer.valid(epoch, full_sort=args.full_sort)
            # evaluate on MRR
            early_stopping(np.array(scores[-1:]), trainer.model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        print("---------------Sample 99 results---------------")
        # load the best model
        trainer.model.load_state_dict(torch.load(args.checkpoint_path,weights_only=True))
        scores, result_info = trainer.test(0, full_sort=args.full_sort)

    print(args_str)
    print(result_info)
    with open(args.log_file, 'a') as f:
        f.write(args_str + '\n')
        f.write(result_info + '\n')

main()
