import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")

### ===== Simplest lost example =====       --------------------------------------------------------------------------------------------------------

output = torch.tensor([[0, 0, 1, 0, 0]], dtype=torch.float32)
target = torch.tensor([2], dtype=torch.long)

loss_function = nn.NLLLoss()

loss = loss_function(output, target)
print("===== Simplest lost example =====")
print(loss)


### ===== With model learning one example =====     -------------------------------------------------------------------------------------------------

print("===== With model learning one example =====")

class NetworkOneData(nn.Module):
    def __init__(self):
        super(NetworkOneData, self).__init__()
        self.forward1 = nn.Linear(5, 10)
        self.forward2 = nn.Linear(10, 5)

    def forward(self, x):
        x = self.forward1(x)
        x = F.relu(x)

        x = self.forward2(x)
        x = F.relu(x)

        output = F.log_softmax(x, dim=1)
        return output

input = torch.tensor([[0, 1, 0, 0, 0]], dtype=torch.float32)
target = torch.tensor([2], dtype=torch.long)

loss_function = nn.NLLLoss()

network = NetworkOneData()
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

print("Start output: ")
output = network(input)
print(output.detach().numpy())

network.train()

for epoch in range(500):
    optimizer.zero_grad()
    output = network(input)

    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Train Epoch: {} Loss: {:.6f}'.format(epoch, loss.item()))
        #print(output.detach().numpy())

print("Final output: ")
print(output.detach().numpy())



### ===== With model learning 2d example =====     -------------------------------------------------------------------------------------------------

print("===== With 2d model learning example =====")

class NetworkOneData(nn.Module):
    def __init__(self):
        super(NetworkOneData, self).__init__()
        self.forward1 = nn.Linear(5 * 2, 20)
        self.forward2 = nn.Linear(20, 10)

    def forward(self, x):
        x = torch.flatten(x, start_dim=0)

        x = self.forward1(x)
        x = F.relu(x)

        x = self.forward2(x)
        x = F.relu(x)
        x = torch.reshape(x, (2, 5))

        output = F.log_softmax(x, dim=1)
        return output

input = torch.tensor([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]], dtype=torch.float32)
target = torch.tensor([2, 3], dtype=torch.long)

loss_function = nn.NLLLoss()

network = NetworkOneData()
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)

print("Start output: ")
output = network(input)
print(output.detach().numpy())

network.train()

for epoch in range(500):
    optimizer.zero_grad()
    output = network(input)

    loss = loss_function(output, target)
    loss.backward()
    optimizer.step()

    if epoch % 100 == 0:
        print('Train Epoch: {} Loss: {:.6f}'.format(epoch, loss.item()))
        #print(output.detach().numpy())

print("Final output: ")
print(output.detach().numpy())