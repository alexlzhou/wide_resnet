import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms

# hyperparameters
epochs = 200
epoch_step = 80
batch = 128
learning_rate = 0.1
learning_rate_decay = 0
learning_rate_decay_ratio = 0.2
weight_decay = 0.0005
dampening = 0
momentum = 0.9
widen_factor = 1
log_interval = 10

random_seed = 1
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

training_data = torchvision.datasets.CIFAR10(
    root='D:/projects_python/datasets/cifar10/',
    train=True,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

testing_data = torchvision.datasets.CIFAR10(
    root='D:/projects_python/datasets/cifar10/',
    train=False,
    download=True,
    transform=torchvision.transforms.ToTensor()
)

train_loader = torch.utils.data.DataLoader(training_data, batch_size=batch, shuffle=True)
test_loader = torch.utils.data.DataLoader(testing_data, batch_size=batch, shuffle=True)

'''
plt.imshow(training_data.__getitem__(0)[0].permute(1, 2, 0))
plt.show()
'''

class WRN(nn.Module):
    def __init__(self):
        super(WRN, self).__init__()
        # nn.SpatialConvolution(3 -> 16, 3x3, 1, 1, 1, 1)
        self.conv1 = nn.Conv2d(3, 16, kernel_size=(3, 3), padding=1, bias=False)
        # nn.SpatialConvolution(16 -> 32, 3x3, 1, 1, 1, 1)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=(3, 3), padding=1, bias=False)
        # nn.SpatialConvolution(16 -> 32, 1x1)
        self.conv3 = nn.Conv2d(16, 32, kernel_size=(1, 1), bias=False)
        # nn.SpatialConvolution(32 -> 32, 3x3, 1, 1, 1, 1)
        self.conv4 = nn.Conv2d(32, 32, kernel_size=(3, 3), padding=1, bias=False)
        # nn.SpatialConvolution(32 -> 64, 3x3, 2, 2, 1, 1)
        self.conv5 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        # nn.SpatialConvolution(32 -> 64, 1x1, 2, 2)
        self.conv6 = nn.Conv2d(32, 64, kernel_size=(1, 1), stride=2, bias=False)
        # nn.SpatialConvolution(64 -> 64, 3x3, 1, 1, 1, 1)
        self.conv7 = nn.Conv2d(64, 64, kernel_size=(3, 3), padding=1, bias=False)
        # nn.SpatialConvolution(64 -> 128, 3x3, 2, 2, 1, 1)
        self.conv8 = nn.Conv2d(64, 128, kernel_size=(3, 3), stride=2, padding=1, bias=False)
        # nn.SpatialConvolution(64 -> 128, 1x1, 2, 2)
        self.conv9 = nn.Conv2d(64, 128, kernel_size=(1, 1), stride=2, bias=False)
        # nn.SpatialConvolution(128 -> 128, 3x3, 1, 1, 1, 1)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=(3, 3), padding=1, bias=False)

        # nn.SpatialBatchNormalization
        self.bn1 = nn.BatchNorm2d(16)
        self.bn2 = nn.BatchNorm2d(32)
        self.bn3 = nn.BatchNorm2d(64)
        self.bn4 = nn.BatchNorm2d(128)

        # nn.SpatialAveragePooling(8x8, 1, 1)
        self.avgpool = nn.AvgPool2d(kernel_size=(8, 8), stride=(1, 1))

        # nn.Linear(128 -> 10)
        self.linear = nn.Linear(128, 10)

    def forward(self, x):
        # nn.CAddTable
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv4(F.relu(self.bn2(self.conv2(x)))) + self.conv3(x)
        # nn.CAddTable
        x = self.conv4(F.relu(self.bn2(self.conv4(F.relu(self.bn2(x)))))) + x
        # nn.CAddTable
        x = F.relu(self.bn2(x))
        x = self.conv7(F.relu(self.bn3(self.conv5(x)))) + self.conv6(x)
        # nn.CAddTable
        x = self.conv7(F.relu(self.bn3(self.conv7(F.relu(self.bn3(x)))))) + x
        # nn.CAddTable
        x = F.relu(self.bn3(x))
        x = self.conv10(F.relu(self.bn4(self.conv8(x)))) + self.conv9(x)
        # nn.CAddTable
        x = self.conv10(F.relu(self.bn4(self.conv10(F.relu(self.bn4(x)))))) + x

        x = F.relu(self.bn4(x))
        x = self.avgpool(x)
        x = x.view(128, -1)
        x = self.linear(x)

network = WRN()
optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

# train the model
train_losses = []
train_counter = []
test_losses = []
test_counter = [i * len(train_loader.dataset) for i in range(epoch + 1)]

def train(epoch):
    network.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = network(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append(
                (batch_idx*64) + ((epoch-1)*len(train_loader.dataset)))
            torch.save(network.state_dict(), 'D:/projects_python/wide_resnet/results/model.pth')
            torch.save(optimizer.state_dict(), 'D:/projects_python/wide_resnet/results/optimizer.pth')


def test():
    network.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = network(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, epochs + 1):
    train(epoch)
    test()