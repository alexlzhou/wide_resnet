from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.datasets
import torchvision.transforms

import config


class WRN(nn.Module):
    def __init__(self, *args):
        super(WRN, self).__init__()
        # nn.SpatialConvolution(3 -> 16, 3x3, 1, 1, 1, 1)
        self.conv1 = nn.Conv2d(3, 16 * args[0], kernel_size=(3, 3), padding=1, bias=False)
        # nn.SpatialConvolution(16 -> 32, 3x3, 1, 1, 1, 1)
        self.conv2 = nn.Conv2d(16 * args[0], 32 * args[0], kernel_size=(3, 3), padding=1, bias=False)
        # nn.SpatialConvolution(16 -> 32, 1x1)
        self.conv3 = nn.Conv2d(16 * args[0], 32 * args[0], kernel_size=(1, 1), bias=False)
        # nn.SpatialConvolution(32 -> 32, 3x3, 1, 1, 1, 1)
        self.conv4_1 = nn.Conv2d(32 * args[0], 32 * args[0], kernel_size=(3, 3), padding=1, bias=False)
        self.conv4_2 = nn.Conv2d(32 * args[0], 32 * args[0], kernel_size=(3, 3), padding=1, bias=False)
        self.conv4_3 = nn.Conv2d(32 * args[0], 32 * args[0], kernel_size=(3, 3), padding=1, bias=False)
        # nn.SpatialConvolution(32 -> 64, 3x3, 2, 2, 1, 1)
        self.conv5 = nn.Conv2d(32 * args[0], 64 * args[0], kernel_size=(3, 3), stride=2, padding=1, bias=False)
        # nn.SpatialConvolution(32 -> 64, 1x1, 2, 2)
        self.conv6 = nn.Conv2d(32 * args[0], 64 * args[0], kernel_size=(1, 1), stride=2, bias=False)
        # nn.SpatialConvolution(64 -> 64, 3x3, 1, 1, 1, 1)
        self.conv7_1 = nn.Conv2d(64 * args[0], 64 * args[0], kernel_size=(3, 3), padding=1, bias=False)
        self.conv7_2 = nn.Conv2d(64 * args[0], 64 * args[0], kernel_size=(3, 3), padding=1, bias=False)
        self.conv7_3 = nn.Conv2d(64 * args[0], 64 * args[0], kernel_size=(3, 3), padding=1, bias=False)
        # nn.SpatialConvolution(64 -> 128, 3x3, 2, 2, 1, 1)
        self.conv8 = nn.Conv2d(64 * args[0], 128 * args[0], kernel_size=(3, 3), stride=2, padding=1, bias=False)
        # nn.SpatialConvolution(64 -> 128, 1x1, 2, 2)
        self.conv9 = nn.Conv2d(64 * args[0], 128 * args[0], kernel_size=(1, 1), stride=2, bias=False)
        # nn.SpatialConvolution(128 -> 128, 3x3, 1, 1, 1, 1)
        self.conv10_1 = nn.Conv2d(128 * args[0], 128 * args[0], kernel_size=(3, 3), padding=1, bias=False)
        self.conv10_2 = nn.Conv2d(128 * args[0], 128 * args[0], kernel_size=(3, 3), padding=1, bias=False)
        self.conv10_3 = nn.Conv2d(128 * args[0], 128 * args[0], kernel_size=(3, 3), padding=1, bias=False)

        # nn.SpatialBatchNormalization
        self.bn1 = nn.BatchNorm2d(16 * args[0])

        self.bn2_1 = nn.BatchNorm2d(32 * args[0])
        self.bn2_2 = nn.BatchNorm2d(32 * args[0])
        self.bn2_3 = nn.BatchNorm2d(32 * args[0])
        self.bn2_4 = nn.BatchNorm2d(32 * args[0])

        self.bn3_1 = nn.BatchNorm2d(64 * args[0])
        self.bn3_2 = nn.BatchNorm2d(64 * args[0])
        self.bn3_3 = nn.BatchNorm2d(64 * args[0])
        self.bn3_4 = nn.BatchNorm2d(64 * args[0])

        self.bn4_1 = nn.BatchNorm2d(128 * args[0])
        self.bn4_2 = nn.BatchNorm2d(128 * args[0])
        self.bn4_3 = nn.BatchNorm2d(128 * args[0])
        self.bn4_4 = nn.BatchNorm2d(128 * args[0])

        # nn.SpatialAveragePooling(8x8, 1, 1)
        self.avgpool = nn.AvgPool2d(kernel_size=(8, 8), stride=(1, 1))

        # nn.Linear(128 -> 10)
        self.linear = nn.Linear(128 * args[0], 10)

    def forward(self, x):
        # nn.CAddTable
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.conv4_1(F.relu(self.bn2_1(self.conv2(x)))) + self.conv3(x)
        # nn.CAddTable
        x = self.conv4_2(F.relu(self.bn2_2(self.conv4_3(F.relu(self.bn2_3(x)))))) + x
        # nn.CAddTable
        x = F.relu(self.bn2_4(x))
        x = self.conv7_1(F.relu(self.bn3_1(self.conv5(x)))) + self.conv6(x)
        # nn.CAddTable
        x = self.conv7_2(F.relu(self.bn3_2(self.conv7_3(F.relu(self.bn3_3(x)))))) + x
        # nn.CAddTable
        x = F.relu(self.bn3_4(x))
        x = self.conv10_1(F.relu(self.bn4_1(self.conv8(x)))) + self.conv9(x)
        # nn.CAddTable
        x = self.conv10_2(F.relu(self.bn4_2(self.conv10_3(F.relu(self.bn4_3(x)))))) + x
        x = F.relu(self.bn4_4(x))

        x = self.avgpool(x)
        x = x.view(-1, 128)
        x = self.linear(x)
        return x


if __name__ == '__main__':
    def train(epoch):
        network.train()
        for batch_idx, curr in enumerate(train_loader):
            data, target = curr[0].to(device), curr[1].to(device)
            optimizer.zero_grad()
            output = network(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()
            if batch_idx % log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset), 100. * batch_idx / len(train_loader),
                    loss.item()))
                train_losses.append(loss.item())
                train_counter.append((batch_idx * 64) + ((epoch - 1) * len(train_loader.dataset)))
                torch.save(network.state_dict(), 'D:/projects_python/_results/wrn/model.pth')
                torch.save(optimizer.state_dict(), 'D:/projects_python/_results/wrn/optimizer.pth')


    def test(k):
        network.eval()
        test_loss = 0
        correct = 0
        with torch.no_grad():
            for curr in test_loader:
                data, target = curr[0].to(device), curr[1].to(device)
                output = network(data)
                test_loss += F.cross_entropy(output, target, size_average=False).item()
                pred = output.data.max(1, keepdim=True)[1]
                correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        test_losses.append(test_loss)
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(test_loss, correct,
                                                                                  len(test_loader.dataset) / k,
                                                                                  100. * correct / len(
                                                                                      test_loader.dataset)))


    # hyperparameters
    n_epochs = config.n_epochs
    epoch_step = config.epoch_step
    n_batch = config.n_batch
    learning_rate = config.learning_rate
    learning_rate_decay = config.learning_rate_decay
    learning_rate_decay_ratio = config.learning_rate_decay_ratio
    weight_decay = config.weight_decay
    dampening = config.dampening
    momentum = config.momentum
    widen_factor = config.widen_factor
    log_interval = config.log_interval

    k_folds = config.k_folds
    # k_folds_results = config.k_folds_results
    random_seed = config.random_seed
    torch.backends.cudnn.enabled = config.cudnn_enabled
    torch.manual_seed(random_seed)

    train_data = torchvision.datasets.CIFAR10(
        root='D:/projects_python/_datasets/cifar10/',
        train=True,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    test_data = torchvision.datasets.CIFAR10(
        root='D:/projects_python/_datasets/cifar10/',
        train=False,
        download=True,
        transform=torchvision.transforms.ToTensor()
    )

    # calculate training set's mean and std
    test_loader_train = torch.utils.data.DataLoader(train_data, batch_size=len(train_data), num_workers=1)

    n_images = 0
    mean_train = 0.
    std_train = 0.
    for batch, _ in test_loader_train:
        batch = batch.view(batch.size(0), batch.size(1), -1)
        n_images += batch.size(0)
        mean_train += batch.mean(2).sum(0)
        std_train += batch.std(2).sum(0)

    mean_train /= n_images
    std_train /= n_images

    del test_loader_train

    # calculate testing set's mean and std
    test_loader_test = torch.utils.data.DataLoader(test_data, batch_size=len(test_data), num_workers=1)

    n_images = 0
    mean_test = 0.
    std_test = 0.
    for batch, _ in test_loader_test:
        batch = batch.view(batch.size(0), batch.size(1), -1)
        n_images += batch.size(0)
        mean_test += batch.mean(2).sum(0)
        std_test += batch.std(2).sum(0)

    mean_test /= n_images
    std_test /= n_images

    del test_loader_test

    # update transform
    train_data.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean_train, std_train
        )
    ])

    test_data.transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(
            mean_test, std_test
        )
    ])

    kfold = KFold(n_splits=k_folds, shuffle=True)
    total_data = torch.utils.data.ConcatDataset([train_data, test_data])

    for fold, (train_ids, test_ids) in enumerate(kfold.split(total_data)):
        train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
        test_subsampler = torch.utils.data.SubsetRandomSampler(test_ids)

        train_loader = torch.utils.data.DataLoader(total_data, batch_size=n_batch, num_workers=2,
                                                   pin_memory=True, sampler=train_subsampler)
        test_loader = torch.utils.data.DataLoader(total_data, batch_size=n_batch, num_workers=2,
                                                  pin_memory=True, sampler=test_subsampler)

        network = WRN()

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        network.to(device)

        optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=momentum)

        # train the model
        train_losses = []
        train_counter = []
        test_losses = []
        test_counter = [i * len(train_loader.dataset) for i in range(n_epochs + 1)]

        for epoch in range(1, n_epochs + 1):
            train(epoch)
            test(kfold.n_splits)
