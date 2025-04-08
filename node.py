import torch
import torchvision
from torchdiffeq import odeint
from torch import nn
import numpy as np


def add_time(in_tensor, t):
    bs, c, w, h = in_tensor.shape
    return torch.cat((in_tensor, t.expand(bs, 1, w, h)), dim=1)


def one_hot_encode(labels):
    one_hot_matrix = torch.zeros(size=(len(labels), 10))
    for i in range(len(labels)):
        one_hot_matrix[i][labels[i]] = 1
    return one_hot_matrix


class ODELayer(nn.Module):
    def __init__(self, num_channels, hidden_dim):
        super(ODELayer, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=num_channels + 1, out_channels=hidden_dim,
                               kernel_size=1, padding=0, stride=1)
        self.activation = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=hidden_dim + 1, out_channels=hidden_dim, kernel_size=3, padding=1, stride=1)
        self.conv3 = nn.Conv2d(in_channels=hidden_dim + 1, out_channels=num_channels,
                               kernel_size=1, padding=0, stride=1)

    def forward(self, t, x):
        xt = add_time(x, t)
        output = self.activation(self.conv1(xt))
        xt = add_time(output, t)
        output = self.activation(self.conv2(xt))
        xt = add_time(output, t)
        return self.conv3(xt)


class ODEBlock(nn.Module):
    def __init__(self, function, tolerance=1e-5):
        super(ODEBlock, self).__init__()
        self.function = function
        self.tol = tolerance

    def forward(self, x, times=torch.tensor([1., 2.])):
        output = odeint(func=self.function, y0=x, t=times, rtol=self.tol, atol=self.tol)
        return output[-1]


class NeuralODE(nn.Module):
    def __init__(self, num_channels=1, hidden_dim=92):
        super(NeuralODE, self).__init__()
        self.initial_velocity = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=hidden_dim, kernel_size=1, padding=0, stride=1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(in_channels=hidden_dim, out_channels=hidden_dim, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.3),
            nn.Conv2d(in_channels=hidden_dim, out_channels=2 * num_channels - 1, kernel_size=1, padding=0, stride=1)
        )
        self.ode_layer = ODEBlock(ODELayer(num_channels, hidden_dim))
        self.output_layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=28 * 28, out_features=10)
        )

    def forward(self, x):
        iv = self.initial_velocity(x)
        ode_output = self.ode_layer(iv)
        model_output = self.output_layer(ode_output)
        return model_output


model = NeuralODE(num_channels=1, hidden_dim=92)

img_std = 0.3081
img_mean = 0.1307

batch_size = 64
train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("data/mnist", train=True, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((img_mean,), (img_std,))
                               ])),
    batch_size=batch_size, shuffle=True
)
print(len(train_loader))

test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST("data/mnist", train=False, download=True,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((img_mean,), (img_std,))
                               ])),
    batch_size=128, shuffle=True
)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
num_epochs = 30
loss = torch.nn.CrossEntropyLoss()
train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []
for epoch in range(num_epochs):
    batch_train_losses = []
    batch_train_accuracies = []
    for images, labels in train_loader:
        model_output = model(images)
        one_hot_labels = one_hot_encode(labels)
        train_loss = loss(model_output, one_hot_labels)
        batch_train_losses.append(train_loss.item())
        print(f"Train Loss: {batch_train_losses[-1]}")
        train_predictions = torch.argmax(model_output, dim=1)
        train_accuracy = torch.sum(train_predictions == labels) / len(labels)
        batch_train_accuracies.append(train_accuracy * 100)
        print(f"Train Accuracy: {batch_train_accuracies[-1]}")
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
    train_losses.append(np.mean(np.array(batch_train_losses)))
    train_accuracies.append(np.mean(np.array(batch_train_accuracies)))
    print(f"Epoch: {epoch}, Train Loss: {train_losses[-1]}, Train Accuracy: {train_accuracies[-1]:03}%")

    batch_test_losses = []
    batch_test_accuracies = []
    for images, labels in test_loader:
        model_output = model(images)
        one_hot_labels = one_hot_encode(labels)
        test_loss = loss(model_output, one_hot_labels)
        batch_test_losses.append(test_loss.item())
        test_predictions = torch.argmax(model_output, dim=1)
        test_accuracy = torch.sum(test_predictions == labels) / len(labels)
        batch_test_accuracies.append(test_accuracy * 100)
    test_losses.append(np.mean(np.array(batch_test_losses)))
    test_accuracies.append(np.mean(np.array(batch_test_accuracies)))
    print(f"Epoch: {epoch}, Test Loss: {test_losses[-1]}, Test Accuracy: {test_accuracies[-1]:03}%")

