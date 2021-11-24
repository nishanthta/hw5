import numpy as np
import scipy.io
from tqdm import tqdm
from nn import *

import torch
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data as data

import torchvision.transforms as transforms
import torchvision.datasets as datasets

class NN(torch.nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
                
        self.input_fc = torch.nn.Linear(in_dim, 256)
        self.hidden_fc = torch.nn.Linear(256, 64)
        self.output_fc = torch.nn.Linear(64, out_dim)
        
    def forward(self, x):
        batch_size = x.shape[0]
        x = x.view(batch_size, -1)
        h1 = F.relu(self.input_fc(x))
        h2 = F.relu(self.hidden_fc(h1))
        y_pred = F.softmax(self.output_fc(h2))
        return y_pred   

train_data = scipy.io.loadmat('./data/nist36_train.mat')
valid_data = scipy.io.loadmat('./data/nist36_valid.mat')



train_x, train_y = torch.from_numpy(train_data['train_data']), torch.from_numpy(train_data['train_labels'])
valid_x, valid_y = torch.from_numpy(valid_data['valid_data']), torch.from_numpy(valid_data['valid_labels'])

mean = train_x.mean() 
std = train_x.std()

train_x -= mean
train_x /= std
valid_x -= mean
valid_x /= std

batch_size = 64
learning_rate = 3e-2
num_epochs = 100
batches = get_random_batches(train_x,train_y,batch_size)

in_dim = train_x.shape[1]
out_dim = 36

model = NN(in_dim, out_dim)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate, momentum=0.9)

train_losses, train_accs, val_losses, val_accs = [], [], [], []

for i in range(num_epochs):
    train_loss, train_acc = 0, 0
    preds, y_npy = [], []
    # model.train()
    # optimizer.zero_grad()

    for xb, yb in batches:
        xb = xb.float()
        yb = yb.long()
        yb = torch.tensor(np.where(yb == 1)[1])

        out = model(xb)
        loss = criterion(out, yb)
        train_loss += loss
        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        preds.extend((torch.argmax(out, axis = 1)).detach().numpy())
        y_npy.extend(yb.detach().numpy())
        _, out_pred = torch.max(out.data, 1)
        train_acc += ((out_pred == yb).sum().item())

    train_losses.append((train_loss*1.) / len(train_x))
    train_accs.append(train_acc / len(train_x))
    print(i, ' ', train_losses[-1], ' ', train_accs[-1])

pass
