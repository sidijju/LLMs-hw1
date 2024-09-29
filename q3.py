import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

class ConvModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(ConvModule, self).__init__()
        
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = nn.ReLU()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x
 
class InceptionModule(nn.Module):
    
    def __init__(self, in_channels, f_1x1, f_3x3):
        super(InceptionModule, self).__init__()
        
        self.branch1 = nn.Sequential(
            ConvModule(in_channels, f_1x1, kernel_size=1, stride=1, padding=0)
        )
        
        self.branch2 = nn.Sequential(
            ConvModule(in_channels, f_3x3, kernel_size=3, stride=1, padding=1)
        )
                
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        return torch.cat([branch1, branch2], 1)

class DownsampleModule(nn.Module):
    def __init__(self, in_channels, f_3x3):
        super(DownsampleModule, self).__init__()
    
        self.branch1 = nn.Sequential(ConvModule(in_channels, f_3x3, kernel_size=3, stride=2, padding=0))
        self.branch2 = nn.MaxPool2d(3, stride=2)
    
    def forward(self, x):
        branch1 = self.branch1(x)
        branch2 = self.branch2(x)
        return torch.cat([branch1, branch2], 1)
   
class InceptionSmall(nn.Module):
    def __init__(self, num_classes = 10):
        super(InceptionSmall, self).__init__()
        
        self.conv1 = ConvModule(in_channels=3,out_channels=96, kernel_size=3, stride=1, padding=0)
        self.inception1 = InceptionModule(in_channels=96,f_1x1=32,f_3x3=32)
        self.inception2 = InceptionModule(in_channels=64,f_1x1=32,f_3x3=48)
        self.down1 = DownsampleModule(in_channels=80,f_3x3=80)
        self.inception3 = InceptionModule(in_channels=160,f_1x1=112,f_3x3=48)
        self.inception4 = InceptionModule(in_channels=160,f_1x1=96,f_3x3=64)
        self.inception5 = InceptionModule(in_channels=160,f_1x1=80,f_3x3=80)
        self.inception6 = InceptionModule(in_channels=160,f_1x1=48,f_3x3=96)   
        self.down2 = DownsampleModule(in_channels=144,f_3x3=96)
        self.inception7 = InceptionModule(in_channels=240,f_1x1=176,f_3x3=160)
        self.inception8 = InceptionModule(in_channels=336,f_1x1=176,f_3x3=160)
        self.meanpool = nn.AdaptiveAvgPool2d((7,7))
        self.fc = nn.Linear(16464, num_classes)
       
    def forward(self, x):
        x = self.conv1(x)
        x = self.inception1(x)
        x = self.inception2(x)
        x = self.down1(x)
        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.down2(x)
        x = self.inception7(x)
        x = self.inception8(x)
        x = self.meanpool(x)
        x = torch.flatten(x,1)
        x = self.fc(x)
        return x
    
class Question3:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.train_data = datasets.FashionMNIST(root='data', train=True, download=True, transform=ToTensor())
        self.test_data = datasets.FashionMNIST(root='data', train=False, download=True, transform=ToTensor())

    def reset_model_parameters(self, model):
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def calculate_accuracy(self, probs, labels):
        predictions = torch.argmax(probs, dim=1)
        correct = (predictions == labels).sum().item()
        total = labels.size(0)
        accuracy = correct / total
        return accuracy
    
    def calculate_val_loss_and_accuracy(self, model, loss_fn, val_dl):
        val_losses = []
        val_accuracies = []
        for _, (val_batch, val_labels) in enumerate(val_dl):
            val_batch, val_labels = val_batch.to(self.device), val_labels.to(self.device)
            val_batch = val_batch.repeat(1, 3, 1, 1)
            val_batch_prediction = model(val_batch)
            val_loss = loss_fn(val_batch_prediction, val_labels)
            val_losses.append(val_loss.item())
            val_accuracies.append(self.calculate_accuracy(val_batch_prediction, val_labels))
        return np.mean(np.array(val_losses)), np.mean(np.array(val_accuracies))

    def part1(self):
        BATCH_SIZE = 64
        N = 5
        model = InceptionSmall().to(self.device)
        train_dl = DataLoader(self.train_data, batch_size = BATCH_SIZE)
        loss_fn = nn.CrossEntropyLoss()
        total_iters = N * len(train_dl)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-09)
        factor = (1e1/1e-9)**(1/total_iters)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=(lambda epoch: factor))

        lrs = []
        training_losses_for_lr = []
        for _ in range(N):
            for _, (batch, labels) in enumerate(train_dl):
                batch, labels = batch.to(self.device), labels.to(self.device)
                batch = batch.repeat(1, 3, 1, 1)

                batch_prediction = model(batch)
                loss = loss_fn(batch_prediction, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                lrs.append(scheduler.get_last_lr())
                training_losses_for_lr.append(loss.item())
        
        plt.figure()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("Learning Rate")
        plt.ylabel("Cross Entropy Loss")
        plt.title("Fashion MNIST")
        plt.plot(lrs, training_losses_for_lr)
        plt.savefig('q3-part1.png')

    def part2(self):
        LR_MIN = 1e-05
        LR_MAX = 1e-01
        BATCH_SIZE = 64
        N = 5
        model = InceptionSmall().to(self.device)
        train_dl = DataLoader(self.train_data, batch_size = BATCH_SIZE)
        val_dl = DataLoader(self.test_data, batch_size = BATCH_SIZE)
        loss_fn = nn.CrossEntropyLoss()

        training_losses_for_iteration = []
        validation_losses_for_iteration = []
        training_accuracy_for_iteration = []
        validation_accuracy_for_iteration = []

        optimizer = torch.optim.Adam(model.parameters(), lr=LR_MIN)
        scheduler = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=LR_MIN, max_lr=LR_MAX)

        iter = 0
        for _ in tqdm(range(N)):
            for _, (batch, labels) in enumerate(train_dl):
                batch, labels = batch.to(self.device), labels.to(self.device)
                batch = batch.repeat(1, 3, 1, 1)

                batch_prediction = model(batch)
                loss = loss_fn(batch_prediction, labels)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                if iter % 100 == 0:
                    val_loss, val_accuracy = self.calculate_val_loss_and_accuracy(model, loss_fn, val_dl)
                    train_loss, train_accuracy = loss.item(), self.calculate_accuracy(batch_prediction, labels)

                    training_losses_for_iteration.append(train_loss)
                    validation_losses_for_iteration.append(val_loss)
                    training_accuracy_for_iteration.append(train_accuracy)
                    validation_accuracy_for_iteration.append(val_accuracy)

                iter += 1
                break
        
        plt.figure()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("Iterations")
        plt.ylabel("Cross Entropy Loss")
        plt.title("Fashion MNIST Losses")
        plt.plot(training_losses_for_iteration, label='train')
        plt.plot(validation_losses_for_iteration, label='val')
        plt.legend()
        plt.savefig('q3-part2-loss.png')

        plt.figure()
        plt.yscale('log')
        plt.xscale('log')
        plt.xlabel("Iterations")
        plt.ylabel("Accuracy")
        plt.title("Fashion MNIST Accuracy")
        plt.plot(training_accuracy_for_iteration, label='train')
        plt.plot(validation_accuracy_for_iteration, label='val')
        plt.legend()
        plt.savefig('q3-part2-accuracy.png')

    def part3(self):
        pass

if __name__ == '__main__':
    q3 = Question3()
    q3.part1()
    #q3.part2()
    #q3.part3()