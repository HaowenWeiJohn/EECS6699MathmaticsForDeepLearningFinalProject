import pickle
import numpy as np
import cv2
import matplotlib.pyplot as plt
from keras.datasets import cifar10, cifar100
from tqdm import tqdm
import Resnet18

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.models import resnet18
import torch
from torch.utils.data import DataLoader, TensorDataset
import config

# define the hyperparameters
from train.utils import test

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batch_size = config.batch_size
learning_rate = config.learning_rate
num_epochs = config.num_epochs
n_classes = config.cifar100_n_classes

# load the CIFAR-10 dataset
(x_train, y_train), (x_test, y_test) = cifar100.load_data()
y_train_onehot = np.zeros((len(y_train), n_classes))
y_test_onehot = np.zeros((len(y_test), n_classes))

y_train_onehot[np.arange(len(y_train)), y_train.flatten()] = 1
y_test_onehot[np.arange(len(y_test)), y_test.flatten()] = 1

y_train = y_train_onehot
y_test = y_test_onehot

x_train = np.swapaxes(x_train, 1, -1)
x_test = np.swapaxes(x_test, 1, -1)

means = np.mean(np.vstack((x_train, x_test)), axis=(0, 2, 3))
stds = np.std(np.vstack((x_train, x_test)), axis=(0, 2, 3))
x_train = (x_train - means.reshape(1, 3, 1, 1)) / stds.reshape(1, 3, 1, 1)
x_test = (x_test - means.reshape(1, 3, 1, 1)) / stds.reshape(1, 3, 1, 1)

x_train_tensor = torch.from_numpy(x_train.astype(np.float32))
y_train_tensor = torch.from_numpy(y_train.astype(np.float32))
x_test_tensor = torch.from_numpy(x_test.astype(np.float32))
y_test_tensor = torch.from_numpy(y_test.astype(np.float32))

train_dataset = TensorDataset(x_train_tensor, y_train_tensor)
test_dataset = TensorDataset(x_test_tensor, y_test_tensor)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# model = ResNet18().to(device)
#
model = Resnet18.ResNet18().to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[150, 250], gamma=0.1)

val_loss_history = []
val_accuracy_history = []


for epoch in range(num_epochs):

    print('Epoch: ', epoch)
    print('learning rate: ', optimizer.param_groups[0]['lr'])

    model.train()

    looper = tqdm(train_dataloader)
    #############################################################################
    for i, (inputs, labels) in enumerate(looper):
        inputs = inputs.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # backward pass and optimizer step
        loss.backward()
        optimizer.step() # update the params

        # update the progress bar
        looper.set_description("Total_loss = %s" % str(loss.item()))
        looper.refresh()

    # update the learning rate
    scheduler.step()

    ##################################################################################################
    avg_loss, accuracy = test(model, test_dataloader, criterion, device)
    print(' ')
    print('val loss: ', avg_loss)
    print('val accuracy: ', accuracy)

    val_loss_history.append(avg_loss)
    val_accuracy_history.append(accuracy)

log_history = {
    "val_loss_history": val_loss_history,
    "val_accuracy_history": val_accuracy_history
}

with open('history.pickle', 'wb') as handle:
    pickle.dump(log_history, handle, protocol=pickle.HIGHEST_PROTOCOL)


