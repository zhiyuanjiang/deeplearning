import torch
from torch.utils.data import Dataset, DataLoader
import os
from PIL import Image
from torchvision import transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import h5py

# Dataset + Dataloader 加载数据是真滴方便

# fuck, 又是学习率没有调好。

# 继承Dataset,实现自己的数据类
class MydataSet(Dataset):
    def __init__(self, root, num, path):
        self.data = [root+str(x)+'.jpg' for x in range(num)]
        with open(path) as file:
            target = file.read()
            target = target.split(',')
            target = [int(x) for x in target]
            self.target = torch.from_numpy(np.array(target)).long()

    def __getitem__(self, index):
        # 读取图片
        image = Image.open(self.data[index])

        # 缩放
        # image = image.resize((1600, 1200), Image.ANTIALIAS)
        # 转换成tensor形式
        return (transforms.ToTensor()(image), self.target[index])
        # 转换成numpy
        # return (np.array(image), self.target[index])

    def __len__(self):
        return len(self.data)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            # define the extracting network here
            # 64*64*3
            nn.Conv2d(3, 32, 7, 1, 3),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            # define the classifier network here
            nn.Linear(32*32*32, 1)
        )

    def forward(self, x):
        # define the forward function here
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        # return F.log_softmax(x)
        return x.sigmoid()

def loss_function(prediction, target):
    # define the loss function here

    target = target.view_as(prediction)
    loss = -target*torch.log(prediction)-(1-target)*torch.log(1-prediction)
    m = loss.shape[0]
    return 1./m*loss.sum()

    # return F.binary_cross_entropy(prediction, target.float())

    # return F.nll_loss(prediction, target, reduction='mean')

def train(train_loader, model, device, optimizer, epoch):
    model.train()
    correct = 0
    val = 0
    for idx, (train_x, train_y) in enumerate(train_loader):
        train_x, train_y = train_x.to(device), train_y.to(device)
        optimizer.zero_grad()
        output = model(train_x)
        # print(output)
        loss = loss_function(output, train_y)
        val = loss.item()
        pred = torch.where(output > 0.5, torch.full_like(output, 1), torch.full_like(output, 0))
        # pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(train_y.view_as(pred)).sum().item()
        loss.backward()
        optimizer.step()
        # if idx != 0 and idx%8 == 0:
        print('epoch '+str(epoch)+' loss is : '+str(loss.item())+' accuracy is : '+str(1.*correct/600))

    return val

def test(test_loader, model, device, epoch):
    model.eval()
    correct = 0
    for idx, (test_x, test_y) in enumerate(test_loader):
        test_x, test_y = test_x.to(device), test_y.to(device)
        output = model(test_x)
        pred = torch.where(output > 0.5, torch.full_like(output, 1), torch.full_like(output, 0))
        # pred = output.max(1, keepdim=True)[1]
        correct += pred.eq(test_y.view_as(pred)).sum().item()
    print('epoch '+str(epoch)+' accuracy is : '+str(1.*correct/150))

def main():

    # mydata = MydataSet('F:\\deeplearning-data\\smile-data\\datasets\\picture\\', 600,
    #                    'F:\\deeplearning-data\\smile-data\\datasets\\train_set_y.txt')
    # mydata_train_loader = DataLoader(mydata, batch_size=600, shuffle=False)
    #
    # mydata = MydataSet('F:\\deeplearning-data\\smile-data\\datasets\\test\\', 150,
    #                    'F:\\deeplearning-data\\smile-data\\datasets\\test_set_y.txt')
    # mydata_test_loader = DataLoader(mydata, batch_size=150, shuffle=False)
    #
    # # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')
    #
    # model = Net().to(device)
    #
    # optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.8)
    #
    # ls = []
    # for epoch in range(50):
    #     val = train(mydata_train_loader, model, device, optimizer, epoch+1)
    #     test(mydata_test_loader, model, device, epoch+1)
    #     ls.append(val)
    #
    # a = [x for x in range(50)]
    # plt.plot(a, ls, 'red')
    # plt.show()
    #
    # torch.save(model.state_dict(), 'F:\\deeplearning-data\\smile-data\\smile-params.pth')

    model = Net()
    model.load_state_dict(torch.load('F:\\deeplearning-data\\smile-data\\smile-params.pth'))
    model.eval()
    image = Image.open('F:\\deeplearning-data\\smile-data\\no-smile3.jpg')
    image = image.resize((64, 64), Image.ANTIALIAS)
    image = transforms.ToTensor()(image)
    data = torch.zeros((1, 3, 64, 64))
    data[0,:,:,:] = image
    out = model(data)
    if out[0].item() > 0.5:
        print('he is smile!')
    else:
        print('he is not smile!')


if __name__ == '__main__':
    main()