import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score

from sklearn.metrics import confusion_matrix

from pytorch_model import MLP

import numpy as np

import joblib

class HOGDataset(Dataset):

    def __init__(self,X,y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, index):
        return self.X[index], self.y[index]
    
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 64

X_test = np.load("X_test.npy")
y_test = np.load("y_test.npy")

scaler = joblib.load("scaler.pkl")

X_test = scaler.transform(X_test)

test_dataset = HOGDataset(X_test,y_test)

test_load = DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

model = MLP().to(device)
model.load_state_dict(torch.load("best_model.pth", map_location=device, weights_only=True))
model.eval()

pos_weight = torch.tensor([31999 / 15260] , dtype=torch.float32)

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

total_loss = 0

all_preds = []
all_labels = []
all_probs = []

with torch.no_grad():

    for X_batch,y_batch in test_load:

        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        outputs =  model(X_batch).squeeze(1)

        loss = criterion(outputs,y_batch)

        total_loss = total_loss + loss.item()

        probs = torch.sigmoid(outputs)
        preds = (probs > 0.5).float()

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y_batch.cpu().numpy())

    total_loss = total_loss / len(test_load)

    accuracy = accuracy_score(all_labels,all_preds)
    precision = precision_score(all_labels,all_preds)
    recall = recall_score(all_labels,all_preds)
    f1 = f1_score(all_labels,all_preds)

    cm = confusion_matrix(all_labels,all_preds)

    print(f"Total loss = {total_loss:.2f}")
    print(f"Accuracy {accuracy:.2f}")
    print(f"Precision {precision:.2f}")
    print(f"Recall {recall:.2f}")
    print(f"f1 score {f1:.2f}")
    print(f"Confusion Matrix")
    print(cm)