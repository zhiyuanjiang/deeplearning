import h5py
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torchvision import transforms

# fuck , 又是learn rate没有调好

# h5文件读取
file = h5py.File('F:\\deeplearning-data\\smile-data\\datasets\\train_happy.h5', 'r')
file1 = h5py.File('F:\\deeplearning-data\\smile-data\\datasets\\test_happy.h5', 'r')
# for key in file.keys():
#     print(key)
#     print(file[key].name)
#     print(file[key].shape)
#     print(type(file[key]))
#     print(file[key].value)
#     print('-----------')

# fuck 注意输入类型

# train_set_x 变成了numpy.ndarray类型 600*64*64*3
train_set_x = file['train_set_x'].value
train_set_y = file['train_set_y'].value
train_set_x = torch.from_numpy(train_set_x).float()
# 调整tensor维度 600*3*64*64
train_set_x = train_set_x.permute(0, 3, 1, 2)
train_set_y = torch.from_numpy(train_set_y).long()
file.close()
#
# # 将tensor转换成图片
# for i in range(600):
#     data = train_set_x[i, :, :, :]
#     picture = transforms.ToPILImage()(data/255.)
#     picture.save('F:\\deeplearning-data\\smile-data\\datasets\\picture\\'+str(i)+'.jpg')

# 保存train_set_y
# file = open('F:\\deeplearning-data\\smile-data\\datasets\\train_set_y.txt', 'w')
# file.write(str(list(train_set_y.numpy())))
# file.close()

# for key in file1.keys():
#     print(key)
#     print(file1[key].name)
#     print(file1[key].shape)
#     print(type(file1[key]))
#     print(file1[key].value)
#     print('-----------')

test_set_x = file1['test_set_x'].value
test_set_y = file1['test_set_y'].value
test_set_x = torch.from_numpy(test_set_x).float()
# 调整tensor维度 150*3*64*64
test_set_x = test_set_x.permute(0, 3, 1, 2)
test_set_y = torch.from_numpy(test_set_y).long()
file1.close()
#
# # # 将tensor转换成图片
# for i in range(150):
#     data = test_set_x[i, :, :, :]
#     picture = transforms.ToPILImage()(data/255.)
#     picture.save('F:\\deeplearning-data\\smile-data\\datasets\\test\\'+str(i)+'.jpg')
#
# # 保存test_set_y
# file = open('F:\\deeplearning-data\\smile-data\\datasets\\test_set_y.txt', 'w')
# file.write(str(list(test_set_y.numpy())))
# file.close()

# print(type(train_set_x))
# print(train_set_x.shape)
# print(type(train_set_y))
# print(train_set_y.shape)

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.features = nn.Sequential(
            # define the extracting network here
            # 64*64*3 --> 64*64*6
            nn.Conv2d(3, 6, 5, 1, 2),
            nn.ReLU(),
            # 64*64*6 --> 32*32*6
            nn.MaxPool2d(2, 2),
            # 32*32*6 --> 28*28*16
            nn.Conv2d(6, 16, 5),
            nn.ReLU(),
            # 28*28*16 --> 14*14*16
            nn.MaxPool2d(2, 2),
            # 14*14*16 --> 10*10*16
            nn.Conv2d(16, 16, 5),
            nn.ReLU(),
            # 10*10*16 --> 5*5*16
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            # define the classifier network here
            nn.Linear(16*5*5, 120),
            nn.ReLU(),
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 2),
        )

    def forward(self, x):
        # define the forward function here
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        x = self.classifier(x)
        return F.log_softmax(x, dim=1)

def loss_function(prediction, target, reduction='mean'):
    # define the loss function here

    # print(prediction.shape)
    # print(prediction.dtype)
    # print(target.shape)
    # print(target.dtype)

    return F.nll_loss(prediction, target, reduction=reduction)

def train(train_x, train_y, model, device, optimizer, batch_size, epoch):
    model.train()
    m = train_x.shape[0]

    num = int(m/batch_size)
    if m%batch_size != 0:
        num = num+1

    for i in range(num):
        s = i*batch_size
        if i == num-1:
            e = m
        else:
            e = (i+1)*batch_size
        data, target = train_x[s:e,:,:,:], train_y[s:e]
        data, target = data.to(device), target.to(device)
        # print('s:'+str(s)+' e:'+str(e)+' data:'+str(data.shape))
        optimizer.zero_grad()
        # print(data.shape)
        output = model(data)
        # print(output.shape)
        loss = loss_function(output, target)
        loss.backward()
        optimizer.step()

        if i != 0 and i%6 == 0:
            print('epoch:'+str(epoch)+' loss:'+str(loss.item()))

def test(test_x, test_y, model, device, epoch):
    model.eval()
    m = test_x.shape[0]
    correct = 0

    data, target = test_x.to(device), test_y.to(device)
    output = model(data)
    pred = output.max(1, keepdim=True)[1]
    correct += pred.eq(target.view_as(pred)).sum().item()

    print('epoch '+str(epoch)+' accuracy is:'+str(1.*correct/m))

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # device = torch.device('cpu')

    model = Net().to(device)

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    for i in range(30):
        train(train_set_x, train_set_y, model, device, optimizer, 64, i+1)
        test(test_set_x, test_set_y, model, device, i+1)


if __name__ == '__main__':
    main()