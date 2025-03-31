
import numpy as np

from scipy.fftpack import fft, ifft
import torch
import torch.nn as nn
from torch.nn.functional import batch_norm
from torchgen.executorch.api.types import tensorListT

"""
x = torch.tensor([1,2,3,4,5,6,7,8,6,9])
#REC  x 中的每个元素都是一个复数，形式为 a + bi，其中 a 是实部，b 是虚部。复数的实部和虚部共同表示了对应频率下的振幅和相位信息。
x = torch.fft.rfft(torch.tensor([1,2,3,4,5,6,7,8,6,9]))
print(x)
sequence_emb_fft = torch.fft.irfft(x, n=10)
print(sequence_emb_fft)
"""



seq_len=10
#x=torch.randn(10)
#[batch_size,seq_len,hideden_size] 2 10 2
#REC  默认沿着输入张量的最后一个维度（时间维度）进行STFT计算[batch_size,hideden_size,seq_len] 2 2 10
x=torch.tensor([[[0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0],
                [1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0]]
                ,
                [[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0],
                 [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]]
                ])
print('x.shape',x.shape)
x =x.reshape(2*2,10)
print(x)
print(x.shape)

#x=torch.tensor([0.0,1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0])

for i in range(1,seq_len//2):
    n_fft = 2*i+1
    weight = torch.ones( 1,2,n_fft//2 + 1, 1, dtype=torch.float32)
    zero_imag = torch.zeros_like(weight)
    #weight = torch.randn(2,n_fft//2 + 1, 1,2, dtype=torch.float32) * 0.02

    weight_complex = torch.stack((weight, zero_imag), dim=-1)

    weight_complex = torch.view_as_complex(weight_complex)
    print('weight.shape1:', weight_complex.shape)
    #print('weight.shape2:',weight.shape)
    print(weight_complex)
    print('n_fft:',n_fft)
    window = torch.ones(n_fft)
    hamming_window = torch.hamming_window(window_length=n_fft, periodic=False)
    pad =  n_fft//2
    #u = nn.functional.pad(x, (pad, pad))
    #print(u)
    u= x
    stft_results = []
    y = torch.stft(u, n_fft=n_fft, hop_length=1,center=True,window=window, return_complex=True)
    print('y.shape:',y.shape)
    y = y.reshape(2,2,n_fft//2+1,10)
    print(y.shape)
    print(y)

    z = y*weight_complex
    print('z.shape:',z.shape)
    z = z.reshape(4,n_fft//2+1,10)
    print(z.shape)
    z = torch.istft(z, n_fft=n_fft, hop_length=1, window=window)
    print('z.shape:', z.shape)
    print('z:   ',z)
"""
y = torch.randn(1, 2,5)
y = y.squeeze(0)
print(y.shape)
tensorList=[]
for i in range(0,5):
    tensorList.append(y)
x =  torch.stack(tensorList, dim=0)
print(x.shape)
"""
    #for j in range(0,x.shape[0]):
     #   hy =
      #  z = torch.istft(y, n_fft=n_fft, hop_length=1, window=hamming_window)
       # print(z)

"""

for i in range(1,seq_len//2):
    n_fft = 2*i+1
    print(n_fft)
    window = torch.ones(n_fft)
    hamming_window = torch.hamming_window(window_length=n_fft, periodic=False)
    pad =  n_fft//2
    u = nn.functional.pad(x, (pad, pad))
    y = torch.stft(u, n_fft=n_fft, hop_length=1, center=False, window=window, return_complex=True)
    print(y.shape)
    z = torch.istft(y, n_fft=n_fft, hop_length=1, window=window)
    print(z)
"""



"""
h = torch.stft(x,n_fft=3,hop_length=1,window = window,return_complex=True)
#生成随机过滤器
weight=torch.randn(y.shape[0],y.shape[1],2,dtype=torch.float32)
weight = torch.view_as_complex(weight)
re=y*weight
re1=h*weight

z = torch.istft(re,n_fft=3,hop_length=1,window = hamming_window)
z1 = torch.istft(re1,n_fft=3,hop_length=1,window = hamming_window)
print(z)
print(z1)
"""
