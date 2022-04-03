import os
import torch
import torch.nn as nn
import torch.utils.data as Data
import torchvision
import matplotlib.pyplot as plt

EPOCH =1 
BATCH_SIZE=50
LR=0.003
DOWNLOAD_MNIST=False

train_data=torchvision.datasets.MNIST(
    root="../mnist/",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=DOWNLOAD_MNIST
)
#print(train_data.train_data.size()) // torch.Size([60000, 28, 28])

#define dataloader 
#弄点老外看不懂的 dataloader 分为若干个小batch，每次抛出来一部分数据
#比如这里每次从dataset拿出50个数据，shuffle是否打乱数据
#1，50，28，28
train_loader= Data.DataLoader(dataset=train_data,
                                batch_size=BATCH_SIZE,
                                shuffle=True
                                )
test_data=torchvision.datasets.MNIST(root="../mnist/",
                                    train=False)
# shape from (2000, 28, 28) to (2000, 1, 28, 28), 
# value in range(0,1)                                   
test_x=torch.unsqueeze(test_data.test_data,dim=1).type(
                            torch.FloatTensor)[:2000]/255.
test_y=test_data.test_labels[:2000]

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
# sequential 用来把网络层和激活层连到一起，输出激活后的网络节点
#           input 1,28,28 output 16,28,28
        self.conv1=nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2
            ),
            nn.ReLU(),#16,28,28
            nn.MaxPool2d(kernel_size=2)#16,14,14 pooling 
        )
        #input 16,14,14,output 16,32,32,这里32是人为规定height，
        #实际多少都行，只要能跟后面匹配上即可
        self.conv2=nn.Sequential(
            nn.Conv2d(16,32,5,1,2),#32,14,14
            nn.ReLU(),#32,14,14
            nn.MaxPool2d(2)#32,7,7
        )
        #FLC 全连接网络， 输出10个数字
        self.out=nn.Linear(32*7*7,10)
    def forward(self,x):
        x=self.conv1(x)
        x=self.conv2(x)
        x=x.view(x.size(0),-1)
        output=self.out(x)
        return output,x
cnn=CNN()
print(cnn)
optimizer=torch.optim.Adam(cnn.parameters(),lr=LR)
#optimizer=torch.optim.SGD(cnn.parameters(),lr=LR) 经测试精度要比adam差很多 需要调lr
loss_func=nn.CrossEntropyLoss()

# training and testing
for epoch in range(EPOCH):
    for step, (b_x, b_y) in enumerate(train_loader):   # gives batch data, normalize x when iterate train_loader

        output = cnn(b_x)[0]               # cnn output
        loss = loss_func(output, b_y)   # cross entropy loss
        optimizer.zero_grad()           # clear gradients for this training step
        loss.backward()                 # backpropagation, compute gradients
        optimizer.step()                # apply gradients

        if step % 100 == 0:#每50次打出来看效果
            test_output, last_layer = cnn(test_x)
            pred_y = torch.max(test_output, 1)[1].data.numpy()
            accuracy = float((pred_y == test_y.data.numpy()).astype(int).sum()) / float(test_y.size(0))
            print('Epoch: ', epoch, '| train loss: %.4f' % loss.data.numpy(), '| test accuracy: %.4f' % accuracy)

test_output, _ = cnn(test_x[:100])
pred_y = torch.max(test_output, 1)[1].data.numpy()
print(pred_y, 'prediction number')
print(test_y[:100].numpy(), 'real number')

