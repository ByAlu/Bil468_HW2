# This code is pulled mostly from the trials at train_pytorch.ipynb

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from sklearn.metrics import confusion_matrix

from pytorch_model import MLP


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")

X_val = np.load("X_val.npy")
y_val = np.load("y_val.npy")

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_val = scaler.fit_transform(X_val)

# print(X_train.mean(), X_train.std())
# print(y_train[:20])

joblib.dump(scaler,"scaler.pkl")

class HOGDataset(Dataset):

    def __init__(self,X,y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

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

num_epochs = 20
best_val_loss = 1000000000

patience = 10
counter = 0

for epoch in range(num_epochs):
    print(f"Epoch {epoch + 1}")

    model.train()
    total_train_loss = 0

    for X_batch, y_batch in train_load:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        #print(y_batch.dtype)

        optimizer.zero_grad()

        outputs = model(X_batch).squeeze(1)

        # probs = torch.sigmoid(outputs)
        # preds = (probs > 0.5).float()

        # print("Outputs",outputs[:10])
        # print("Probabilities", probs[:10])
        # print("Labels",y_batch[:10])

        # acc = (preds == y_batch).float().mean()
        # print("Batch accuracy", acc.item())

        loss = criterion(outputs, y_batch)
        loss.backward()

        #print(loss.item())

        #print("Gradient mean:",model.network[0].weight.grad.abs().mean())

        optimizer.step()

        total_train_loss = total_train_loss + loss.item()
    
    total_train_loss = total_train_loss / len(train_load)

    print(f"Epoch {epoch + 1}: Total train loss = {total_train_loss:.2f}")

    # Now we evaulate using val data

    model.eval()

    total_val_loss = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():

        for X_batch, y_batch in val_load:

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            outputs = model(X_batch).squeeze(1)

            loss = criterion(outputs, y_batch)

            total_val_loss = total_val_loss + loss.item()

            probs = torch.sigmoid(outputs)
            preds = (probs > 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())

    total_val_loss = total_val_loss / len(val_load)


    accuracy = accuracy_score(all_labels,all_preds)
    precision = precision_score(all_labels,all_preds)
    recall = recall_score(all_labels,all_preds)
    f1 = f1_score(all_labels,all_preds)

    cm = confusion_matrix(all_labels,all_preds)

    print(f"Epoch {epoch + 1}: Total val loss = {total_val_loss:.2f}")
    print(f"Accuracy {accuracy:.2f}")
    print(f"Precision {precision:.2f}")
    print(f"Recall {recall:.2f}")
    print(f"f1 score {f1:.2f}")
    print(f"Confusion Matrix")
    print(cm)

    print

    if total_val_loss < best_val_loss:

        best_val_loss = total_val_loss

        counter = 0

        torch.save(model.state_dict(),"best_model.pth")
    
    else:
        counter = counter + 1

    if counter >= patience:
        break
