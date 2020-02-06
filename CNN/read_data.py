import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
import numpy as np
from torchvision import transforms

# Dataset + Dataloader 加载数据是真滴方便

# 继承Dataset,实现自己的数据类
class MydataSet(Dataset):
    def __init__(self, root):
        self.data = [root+'\\'+it for it in os.listdir(root) if it.endswith('.jpg')]

    def __getitem__(self, index):
        # 读取图片
        image = Image.open(self.data[index])
        # 缩放
        image = image.resize((1600, 1200), Image.ANTIALIAS)
        # 转换成tensor形式
        return transforms.ToTensor()(image)

    def __len__(self):
        return len(self.data)

# mydata = MydataSet('f:\\wallpaper')
# a = mydata.__getitem__(0)
# print(type(a))
# print(a.shape)

mydata = MydataSet('f:\\wallpaper')
mydata_train_loader = DataLoader(mydata, batch_size=2, shuffle=False)
print(len(mydata_train_loader))
for idx, data in enumerate(mydata_train_loader):
    print(idx)
    print(type(data))
    print(data.shape)

