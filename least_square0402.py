import torch 
import matplotlib.pyplot as plt
import torch.nn.functional as F
#produce number 100 x dim=1 
x=torch.unsqueeze(torch.linspace(-1,1,100),dim=1)
y=x.pow(2)+torch.rand(x.size())

plt.scatter(x,y)
# plt.show()
class Net(torch.nn.Module):
    def __init__(self,n_feature,n_hidden,n_output):
        super().__init__()
        self.hidden=torch.nn.Linear(n_feature,n_hidden)
        self.predict=torch.nn.Linear(n_hidden,n_output)
    def forward(self,x):
        x=F.relu(self.hidden(x))
        x=self.predict(x)
        return x

net=Net(1,10,1)
print(net)

optimizer=torch.optim.SGD(net.parameters(),lr=0.49)
loss_func=torch.nn.MSELoss()

