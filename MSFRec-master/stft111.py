import copy
import math
from distutils.core import setup_keywords
from functools import partial

import torch
import torch.nn as nn
from pandas.core.methods.describe import select_describe_func


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class STFTLayer(nn.Module):
    def __init__(self, args):
        super(STFTLayer, self).__init__()
        self.stft_type = args.stft_type
        self.stft_weights = nn.ModuleDict()

        if self.stft_type >= 1:
            self.stft_weight3 = nn.Parameter(
                torch.randn(1,args.hidden_size, 3 // 2 + 1, 1, 2, dtype=torch.float32) * 0.02
            )

        if self.stft_type>=2:
            self.stft_weight5 = nn.Parameter(
                torch.randn(1,args.hidden_size, 5 // 2 + 1, 1, 2, dtype=torch.float32) * 0.02
            )
        if self.stft_type>=3:
            self.stft_weight7 = nn.Parameter(
                torch.randn(1,args.hidden_size, 7 // 2 + 1, 1, 2, dtype=torch.float32) * 0.02
            )
        if self.stft_type>=4:
            self.stft_weight9 = nn.Parameter(
                torch.randn(1,args.hidden_size, 9 // 2 + 1, 1, 2, dtype=torch.float32) * 0.02
            )

        self.complex_weight = nn.Parameter(
            torch.randn(1, args.max_seq_length // 2 + 1, args.hidden_size, 2, dtype=torch.float32) * 0.02)
        self.alpha = nn.Parameter(torch.tensor(0.5, dtype=torch.float32))
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12 )


    def forward(self, input_tensor):
        # [batch, seq_len, hidden]

        batch, seq_len, hidden = input_tensor.shape
        #REC [batch, hidden, seq_len]
        #REC 默认沿着输入张量的最后一个维度（时间维度）进行STFT计算
        for i in range(1,self.stft_type+1):
            n_fft = 2*i+1
            pad = n_fft // 2
            u = nn.functional.pad(input_tensor.permute(0,2,1), (pad, pad))
            #REC torch.ones 默认在 CPU 上创建张量
            window = torch.ones(n_fft,device = input_tensor.device)
            #window = torch.hamming_window(window_length=n_fft, device = input_tensor.device)
            if n_fft==3:
                weight = torch.view_as_complex(self.stft_weight3)
            elif n_fft==5:
                weight = torch.view_as_complex(self.stft_weight5)
            elif n_fft==7:
                weight = torch.view_as_complex(self.stft_weight7)
            elif n_fft==9:
                weight = torch.view_as_complex(self.stft_weight9)
            len=u.shape[2]
            u = u.reshape(batch*hidden,len)
            x = torch.stft(u, n_fft=n_fft, hop_length=1,center=False,window=window, return_complex=True)
            x = x.reshape(batch,hidden,n_fft//2+1,seq_len)
            x = x * weight
            x = x.reshape(batch*hidden,n_fft//2+1,seq_len)
            z = torch.istft(x, n_fft=n_fft, hop_length=1, window=window)
            out = z.reshape(batch,hidden,seq_len).permute(0,2,1)
            out = self.LayerNorm(out)
            if i==1:
                sequence_emb = out
            else:
                sequence_emb = sequence_emb + out

        x_fft = torch.fft.rfft(input_tensor, dim=1, norm='ortho')
        weight = torch.view_as_complex(self.complex_weight)
        x_fft = x_fft * weight
        sequence_fft = torch.fft.irfft(x_fft, n=seq_len, dim=1, norm='ortho')
        sequence_fft = self.LayerNorm(sequence_fft)
        sequence_emb = self.alpha*sequence_emb + sequence_fft*(1-self.alpha)


        hidden_states = self.out_dropout(sequence_emb)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


