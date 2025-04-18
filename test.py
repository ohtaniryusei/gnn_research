import argparse
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

from full_model import FullModel
from utils.data_loader import load_transaction_batch

def test(model_name):
    print(f"🚀 モデル = {model_name} でAUC評価を開始...")

    # 設定
    config = {
        'in_dim': 128,
        'time_config': (16, 16, 8, 32, 128),
        'dynamic_type': model_name
    }

    model = FullModel(config)
    model.eval()  # 評価モード

    aucs = []
    criterion = nn.BCELoss()

    for i in range(10):  # 10バッチで平均
        with torch.no_grad():
            h_static, h_neighbors, t, t_prime, sem_feat, y_true = load_transaction_batch()
            y_pred = model(h_static, h_neighbors, t, t_prime, sem_feat).squeeze()

            auc = roc_auc_score(y_true.numpy(), y_pred.numpy())
            loss = criterion(y_pred, y_true).item()

            aucs.append(auc)
            print(f"Batch {i+1:02d} | AUC: {auc:.4f} | Loss: {loss:.4f}")

    print(f"\n✅ 平均AUC ({model_name}): {sum(aucs)/len(aucs):.4f}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, choices=['tgat', 'graphmixer'], default='tgat',
                        help='動的GNNモデルの種類（tgat or graphmixer）')
    args = parser.parse_args()

    test(args.model)
