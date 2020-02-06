import numpy as np

def zero_pad(A, pad):
    return np.pad(A, ((0, 0), (pad, pad), (pad, pad), (0, 0)), 'constant', constant_values=0)

def conv_single_step(a_slice_prev, w, b):
    s = a_slice_prev*w+b
    return np.sum(s)

def conv_forward(A_prev, W, b, hparameters):
    # m samples
    (m, pre_n_H, pre_n_W, pre_n_C) = A_prev.shape
    # n filters
    (n_filter, n_f, n_f, pre_n_C) = W.shape

    pad = hparameters['pad']
    stride = hparameters['stride']

    n_H = int((pre_n_H+2*pad-n_f)/stride)+1
    n_W = int((pre_n_W+2*pad-n_f)/stride)+1
    n_C = n_filter

    A_prev_pad = zero_pad(A_prev, pad)
    Z = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        t = A_prev_pad[i]
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):
                    s_H = h*stride
                    e_H = s_H+n_f
                    s_W = w*stride
                    e_W = s_W+n_f

                    Z[i, h, w, i] = conv_single_step(t[s_H:e_H, s_W:e_W, :], W[c,:,:,:], b[0, c])

    return Z

def pool_forward(A_prev, hparameters, type='max'):

    (m, pre_n_H, pre_n_W, pre_n_C) = A_prev.shape

    n_f = hparameters['f']
    stride = hparameters['stride']

    n_H = int((pre_n_H-n_f)/stride)+1
    n_W = int((pre_n_W-n_f)/stride)+1
    n_C = pre_n_C

    Z = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        for h in range(n_H):
            for w in range(n_W):
                for c in range(n_C):

                    s_H = h*stride
                    e_H = s_H+n_f
                    s_W = w*stride
                    e_W = s_W+n_f

                    if type == 'max':
                        Z[i, h, w, c] = np.max(A_prev[i, s_H:e_H, s_W:e_W, c])


    return Z

import torch
import numpy as np
#### build tensor
a = torch.Tensor([[1, 2, 3], [4, 5, 6]])   # the same as torch.FloatTensor, like np.array()
# print(a[1, 1])

#### 初始化
# 均匀分布
a = torch.rand(2, 3)
# 标准正态分布
a = torch.randn(2, 3)
# 全0的tensor
a = torch.zeros(2, 3)
# 生成和a一样大小的全零tensor
b = torch.zeros_like(a)

a = torch.arange(start=1, end=10, step=1)

a = torch.full((2, 3), 1)

# print(a)

# 改变tensor形状
a = torch.arange(0, 10, 1)
print(a.view(2, 5))
b = a.reshape(2, 5)
c = a.view_as(b)
# print(a)
print(c)

a = torch.full((3, 2), 0)
print(a.shape[0])

# 调整tensor维度位置
a = torch.randn(2, 3)
print(a)
b = a.permute(1, 0)
print(b)

# torch.where()
a = torch.randn(3, 3)
b = torch.where(a > 0, torch.full_like(a, 1), torch.full_like(a, 0))
print(a)
print(b)

####### transforms.ToTensor()(image)
from PIL import Image
from torchvision import transforms
image = Image.open('F:\\deeplearning-data\\smile-data\\datasets\\test\\0.jpg')
# image.show()
print(transforms.ToTensor()(image))
print(np.array(image))

############### torch.sigmoid()
import math
a = torch.Tensor([2])
print(a.sigmoid())
print(1./(1.+math.exp(-2)))

############## np.argmax()
import numpy as np
a = np.array([[2,3,4],[1,2,0]])
print(a)
print(np.argmax(a, axis=1))

a = np.random.randn(4, 3, 2, 2)
print(a)
print(np.argmax(a, axis=-1))

a = np.array([1, 2, 3])
print(a > 2)