# This code is pulled mostly from the trials at train_pytorch.ipynb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from pytorch_model import MLP

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)

joblib.dump(scaler,"scaler.pkl")

class HOGDataset(Dataset):

    def __init__(self,X,y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(X, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
train_dataset = HOGDataset(X_train,y_train)
train_load = DataLoader(train_dataset,batch_size=64,shuffle=True)

val_dataset = HOGDataset(X_val,y_val)
val_load = DataLoader(val_dataset,batch_size=64,shuffle=False)

# Below is new code differing from the ipynb file

model = MLP()

optimizer = optim.Adam(model.parameters(),lr=1e-3)

# weight numbers pulled from the ipynb tests
pos_weight = torch.tensor([31999 / 15260] , dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

model = model.to(device)

# Train epochs

num_epochs = 100

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")

    model.train()
    total_train_loss = 0

    for X_batch, y_batch in train_load:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        optimizer.zero_grad()

        outputs = model(X_batch).squeeze(1)

        loss = criterion(outputs, y_batch)
        loss.backward()

        optimizer.step()

        total_train_loss = total_train_loss + loss
    
    total_train_loss = total_train_loss / len(train_load)

    print(f"Epoch {epoch + 1}: Total loss = {total_train_loss:.2f}")