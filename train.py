import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_auc_score

from full_model import FullModel
from utils.data_loader import load_transaction_batch

def train():
    config = {
        'in_dim': 128,
        'time_config': (16, 16, 8, 32, 128),  # d_re, d_ab, d_se, d_time, hidden
    }

    model = FullModel(config)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.BCELoss()

    for epoch in range(5):
        model.train()
        h_static, h_neighbors, t, t_prime, sem_feat, y_true = load_transaction_batch()

        y_pred = model(h_static, h_neighbors, t, t_prime, sem_feat).squeeze()
        loss = criterion(y_pred, y_true)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        auc = roc_auc_score(y_true.detach().numpy(), y_pred.detach().numpy())
        print(f"Epoch {epoch+1} | Loss: {loss.item():.4f} | AUC: {auc:.4f}")

if __name__ == '__main__':
    train()
