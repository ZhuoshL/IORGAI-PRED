
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset
from utils import train_val_test_split
from early_stopping import EarlyStopping
from torch.optim import lr_scheduler
from model import LSTMAttentionRegressor
import os
from scipy.stats import pearsonr



def run(model):
    train_losses, val_losses = [], []

    early_stop = EarlyStopping(patience=20, path=model2save + '.pth', verbose=True)
    for epoch in range(num_epochs):
        for param_group in optimizer.param_groups:
            print("LR", param_group['lr'])
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            pred = model(xb)
            loss = loss_fn(pred, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)

        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb)
                loss = loss_fn(pred, yb)
                total_val_loss += loss.item() * xb.size(0)
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

        scheduler.step()

        early_stop(avg_val_loss, model)
        if early_stop.early_stop:
            print("EarlyStopping")
            break
    model.load_state_dict(torch.load(model2save + '.pth'))
    model.eval()
    all_pred = []
    test_id = []
    test_id.append(te_index.reshape(-1, 1).astype(str))
    test_label, test_pred = [],  []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            pred = model(xb)
            test_pred.extend(pred.cpu().detach().numpy())
            test_label.extend(yb.cpu().detach().numpy())
    test_id.append(np.array(test_label))
    test_id.append(np.array(test_pred))
    test_id = np.concatenate(test_id, axis=1)
    return train_losses, val_losses, test_id


base_dir = "./datasets/test_result/stats/"
feature = np.load(base_dir + "all_feature_stats.npy", allow_pickle=True).astype(np.float32)
label = np.load(base_dir + "all_label.npy", allow_pickle=True).astype(np.float32)

# 转为 Tensor
feature_tensor = torch.tensor(feature).to(torch.float32)

label_tensor = torch.tensor(label).to(torch.float32)

save_dir = './regresult/'
os.makedirs(save_dir, exist_ok=True)

num_subj = feature_tensor.shape[0]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 拆分训练/验证
all_test_result = []
all_train_loss = []
all_val_loss = []
kfold = 5
for fold in range(0, kfold):
    print("=========================================\n")
    print('FOLD: ' + str(fold) + '\n')
    model2save = save_dir + 'regmodel_' + str(fold)
    result2save = save_dir + 'regtestresult_' + str(fold)
   
    # 2️⃣ 获取索引（假设你自己实现了 train_val_test_split）
    tr_index, val_index, te_index = train_val_test_split(kfold=kfold, fold=fold, num=num_subj, randomseed=555)

    train_feature = feature_tensor[tr_index]
    # 标准化
    mean_feature = train_feature.mean(axis=(0,1))
    std_feature = train_feature.std(axis=(0,1))
    std_feature = std_feature.clamp_min(1e-8)
    normalized_feature = (feature_tensor - mean_feature) / std_feature 
    dataset = TensorDataset(normalized_feature, label_tensor)


    # 3️⃣ 用 Subset 按索引划分
    train_ds = Subset(dataset, tr_index)
    val_ds = Subset(dataset, val_index)
    test_ds = Subset(dataset, te_index)

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

    model = LSTMAttentionRegressor(input_dim=feature_tensor.shape[-1], hidden_dim=256, num_layers=4, output_dim=1, dropout=0.1)

    print(
        "Total number of paramerters in networks is {}  ".format(sum(x.nelement() for x in model.parameters())))

    # ============================
    # 3️⃣ 训练准备
    # ============================

    model.to(device)

    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)
    # ============================
    # 4️⃣ 训练循环
    # ============================
    num_epochs = 1000
    train_losses, val_losses, test_id = run(model)
    test_D7_mse = np.mean((test_id[:, 1].astype(np.float32) - test_id[:, 2].astype(np.float32)) ** 2)
    print("=================Fold: " + str(fold) + "==================")
    print("result: \nD7_mse: " + str(test_D7_mse))

    all_test_result.append(test_id)
    all_train_loss.append(train_losses)
    all_val_loss.append(val_losses)

all_test_result = np.concatenate(all_test_result, 0)
df = pd.DataFrame(all_test_result, columns=["编号", "D7_true", "D7_pred"])
df.to_csv(save_dir + "reg_output2.csv", index=False)
# 保存为 CSV
D7_mse = np.mean((all_test_result[:, 1].astype(np.float32) - all_test_result[:, 2].astype(np.float32)) ** 2)
D7_mae = np.mean(np.abs(all_test_result[:, 1].astype(np.float32) - all_test_result[:, 2].astype(np.float32)))
print("====================================================")
print("\nall_result: \nD7_mse: " + str(D7_mse))
print("D7_mae: " +  str(D7_mae))
r, p = pearsonr(all_test_result[:, 1].astype(np.float32), all_test_result[:, 2].astype(np.float32))
print("pearson：" + str(r))