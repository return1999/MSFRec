

class FMLPRecModel(nn.Module):
    def __init__(self, args):
        #REC 调用父类的构造函数
        super(FMLPRecModel, self).__init__()
        self.args = args
        #REC 获得项目的嵌入  参数有 项目的大小 嵌入的长度 填充标记的向量用0表示
        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0)
        #REC 获得位置的嵌入
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)
        #REC 层次归一化
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        #REC Dropout正则化
        self.dropout = nn.Dropout(args.hidden_dropout_prob)
        self.item_encoder = Encoder(args)
        #REC 初始化所有模型权重
        self.apply(self.init_weights)
    #REC 获得交互序列的组合嵌入
    def add_position_embedding(self, sequence):
        #REC 如果你有一个形状为 [batch_size, seq_length] 的 sequence 张量
        #REC sequence.size(1)返回的是seq_length
        seq_length = sequence.size(1)
        #REC 生成一个从 0 开始到 seq_length（不包括 seq_length）的整数序列。这个序列将用作位置索引。
        #REC position_ids: tensor([0, 1, 2, 3, 4])
        position_ids = torch.arange(seq_length, dtype=torch.long, device=sequence.device)
        #REC unsqueeze将一维张量变为二维张量 # position_ids: tensor([[0, 1, 2, 3, 4]])
        #REC expand_as(sequence) 将 position_ids 扩展为与 sequence 相同的形状
        #REC 假设sequence是[3, 5] 那么position_ids: tensor([[0, 1, 2, 3, 4],[0, 1, 2, 3, 4],[0, 1, 2, 3, 4]])
        position_ids = position_ids.unsqueeze(0).expand_as(sequence)
        #REC 获得项目嵌入 每个项ID都会被映射到一个预先训练好或者随机初始化的向量
        #REC item_embeddings维度为[batch_size, seq_length, embedding_dim]
        item_embeddings = self.item_embeddings(sequence)
        #REC 获得位置嵌入 position_embeddings维度为[batch_size, seq_length, embedding_dim]
        position_embeddings = self.position_embeddings(position_ids)
        #REC 将位置嵌入和序列嵌入加起来 获得感知位置的嵌入
        sequence_emb = item_embeddings + position_embeddings
        sequence_emb = self.LayerNorm(sequence_emb)
        sequence_emb = self.dropout(sequence_emb)

        return sequence_emb

    # same as SASRec
    def forward(self, input_ids):
        #REC 创建一个和input_ids维度一致的二进制掩码 非填充为True  填充为False
        #REC attention_mask维度为[batch_size, seq_len]
        attention_mask = (input_ids > 0).long()
        #REC [batch_size, 1, 1, seq_len]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2) # torch.int64
        #REC 返回seq_len
        max_len = attention_mask.size(-1)
        attn_shape = (1, max_len, max_len)
        #REC 创建一个上三角矩阵 实现掩码注意力机制的效果
        #REC torch.ones(attn_shape)：填充1的张量 维度为attn_shape
        #REC torch.triu(input, diagonal=1)：只保留主对角线上面的值，对角线和主对角线下方的元素都变成0
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1) # torch.uint8
        #REC 下三角 包含主对角线  subsequent_mask维度为(1，1, max_len, max_len)
        subsequent_mask = (subsequent_mask == 0).unsqueeze(1)
        #REC 数据类型转换成long
        subsequent_mask = subsequent_mask.long()

        #REC 是否使用GPU
        if self.args.cuda_condition:
            #REC subsequent_mask 张量移动到 GPU 上。
            subsequent_mask = subsequent_mask.cuda()
        #REC 张量逐元素相乘
        #REC 将序列中的填充值和下三角进行求and操作，得到每个矩阵的可用值（去除填充值和未来的值）
        #REC extended_attention_mask 维度为(batch_size，1, max_len, max_len)
        extended_attention_mask = extended_attention_mask * subsequent_mask
        #REC next(self.parameters()).dtype 获取模型第一个参数的数据类型
        #REC 将extended_attention_mask转换为与模型参数的数据类型一致
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        #REC 张量中的0值变-10000,1值变0
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        #REC 获取序列的嵌入
        sequence_emb = self.add_position_embedding(input_ids)

        item_encoded_layers = self.item_encoder(sequence_emb,
                                                extended_attention_mask,
                                                output_all_encoded_layers=True,
                                                )

        sequence_output = item_encoded_layers[-1]

        return sequence_output
    #REC 初始化权重
    def init_weights(self, module):
        """ Initialize the weights.
        """
        #REC 检查模块是不是全连接层或嵌入层
        if isinstance(module, (nn.Linear, nn.Embedding)):
            # Slightly different from the TF version which uses truncated_normal for initialization
            # cf https://github.com/pytorch/pytorch/pull/5617
            #REC 使用正态分布初始化该层的权重
            module.weight.data.normal_(mean=0.0, std=self.args.initializer_range)
        #REC 检查模块是不是归一化层
        elif isinstance(module, LayerNorm):
            #REC 将该层的偏置（bias）初始化为0，权重（weight）初始化为1
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        #REC 检查模块是不是全连接层，并且该层有偏置项
        if isinstance(module, nn.Linear) and module.bias is not None:
            #REC 将该层的偏置（bias）初始化为0
            module.bias.data.zero_()
