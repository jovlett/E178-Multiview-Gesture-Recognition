import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
'''
# KEEP SEED ONLY FOR TESTING
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)'''

# Load dataset
df = pd.read_csv("data/normalised_hand_data_with_clusters.csv")

# Features = all columns except last, video_id, and frame_id
X = df.drop(columns=["video_id", "frame_id", "Cluster_Number"]).values

# Labels = last column
y = df.iloc[:, -1].values.astype(int)

'''
print("X shape:", X.shape)
print("y shape:", y.shape)
print("Labels:", np.unique(y))

print(len(df.columns))
print(df.columns)'''

# Split into train and validation sets
Xtrain, Xvalid, ytrain, yvalid = train_test_split(X, y, test_size=0.99, stratify=y) # 80% training, 20% test

# Scale features
scaler = StandardScaler()

Xtrain = scaler.fit_transform(Xtrain)
Xvalid = scaler.transform(Xvalid)

# Convert to PyTorch tensors
Xtrain_t = torch.tensor(Xtrain, dtype=torch.float32)
Xvalid_t = torch.tensor(Xvalid, dtype=torch.float32)

# For CrossEntropyLoss, labels must be LongTensor, shape (N,)
ytrain_t = torch.tensor(ytrain, dtype=torch.long)
yvalid_t = torch.tensor(yvalid, dtype=torch.long)


train_loader = DataLoader(
    TensorDataset(Xtrain_t, ytrain_t),
    batch_size=32,
    shuffle=True
)

D = Xtrain.shape[1]   # number of input features
num_classes = 5       # labels are 0, 1, 2, 3, 4

model_mlp = nn.Sequential(
    nn.Linear(D, 64),
    nn.ReLU(),
    nn.Linear(64, 32),
    nn.ReLU(),
    nn.Linear(32, num_classes)
)

for layer in model_mlp:
    if isinstance(layer, nn.Linear):
        nn.init.xavier_uniform_(layer.weight)
        nn.init.zeros_(layer.bias)

# Train model
loss_fn = nn.CrossEntropyLoss()
optimizer = optim.Adam(model_mlp.parameters(), lr=0.001)
history = {"epoch": [], "train_acc": [], "val_acc": []}

for epoch in range(20):
    print(f"Starting epoch {epoch}")
    model_mlp.train()

    for xb, yb in train_loader:
        optimizer.zero_grad()

        logits = model_mlp(xb)
        loss = loss_fn(logits, yb)

        loss.backward()
        optimizer.step()
    '''
    model_mlp.eval()
    with torch.no_grad():
        train_logits = model_mlp(Xtrain_t)
        valid_logits = model_mlp(Xvalid_t)

        train_pred = torch.argmax(train_logits, dim=1).numpy()
        valid_pred = torch.argmax(valid_logits, dim=1).numpy()

    train_acc = accuracy_score(ytrain, train_pred)
    val_acc = accuracy_score(yvalid, valid_pred)

    history["epoch"].append(epoch)
    history["train_acc"].append(train_acc)
    history["val_acc"].append(val_acc)

    print(f"Epoch {epoch}: train acc = {train_acc:.4f}, val acc = {val_acc:.4f}")'''

    if epoch % 5 == 0:
        model_mlp.eval()
        with torch.no_grad():
            train_logits = model_mlp(Xtrain_t)
            valid_logits = model_mlp(Xvalid_t)

            train_pred = torch.argmax(train_logits, dim=1).numpy()
            valid_pred = torch.argmax(valid_logits, dim=1).numpy()

        train_acc = accuracy_score(ytrain, train_pred)
        val_acc = accuracy_score(yvalid, valid_pred)

        print(f"Epoch {epoch}: train acc = {train_acc:.4f}, val acc = {val_acc:.4f}")


# EVALUATION

print("\nFinal validation accuracy:", accuracy_score(yvalid, valid_pred))

print("\nConfusion matrix:")
print(confusion_matrix(yvalid, valid_pred))

print("\nClassification report:")
print(classification_report(yvalid, valid_pred))