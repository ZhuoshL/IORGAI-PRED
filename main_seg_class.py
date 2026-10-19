
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset, random_split, Subset, WeightedRandomSampler
from utils import train_val_test_split
from early_stopping import EarlyStopping
from torch.optim import lr_scheduler
from model_class import LSTMAttentionClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score, roc_curve, auc
from sklearn.preprocessing import label_binarize
from tqdm import tqdm
import os
import torch.nn.functional as F


class CombinedLoss(nn.Module):
    """
    Focal Loss + Label Smoothing CrossEntropy
    适用于多分类（你的三分类任务直接用）
    """

    def __init__(self, num_classes=3, alpha=None, gamma=2.0, smoothing=0.1):
        super(CombinedLoss, self).__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.smoothing = smoothing

        if alpha is not None:
            self.alpha = torch.tensor(alpha, dtype=torch.float32)
        else:
            self.alpha = None

    def forward(self, logits, target):
        """
        logits: (B, C)
        target: (B,)
        """

        # ---------- Label Smoothing ----------
        log_probs = F.log_softmax(logits, dim=1)

        with torch.no_grad():
            true_dist = torch.zeros_like(log_probs)
            true_dist.fill_(self.smoothing / (self.num_classes - 1))
            true_dist.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)

        ce_loss = torch.sum(-true_dist * log_probs, dim=1)

        # ---------- Focal Loss ----------
        pt = torch.exp(-ce_loss)  # 相当于 softmax 概率
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        # ---------- 类别权重 ----------
        if self.alpha is not None:
            alpha = self.alpha.to(logits.device)
            focal_loss = alpha[target] * focal_loss

        return focal_loss.mean()

# ============================
# 2️⃣ 定义 LSTM 模型
# ============================

# 实例化模型
def run(model, train_loader, val_loader, test_loader, optimizer, loss_fn, scheduler, num_epochs=1000, device='cpu'):
    train_losses, val_losses = [], []
    early_stop = EarlyStopping(patience=20, path=model2save + '.pth', verbose=True)
    for epoch in range(num_epochs):
        # 训练
        model.train()
        total_train_loss = 0
        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device).long()
            optimizer.zero_grad()
            logits = model(xb)
            loss = loss_fn(logits, yb)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item() * xb.size(0)
        avg_train_loss = total_train_loss / len(train_loader.dataset)
        train_losses.append(avg_train_loss)

        # 验证
        model.eval()
        total_val_loss = 0
        val_true, val_pred = [], []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb, yb = xb.to(device), yb.to(device).long()
                logits = model(xb)
                loss = loss_fn(logits, yb)
                total_val_loss += loss.item() * xb.size(0)
                val_pred.extend(torch.argmax(logits, dim=1).cpu().numpy())
                val_true.extend(yb.cpu().numpy())
        avg_val_loss = total_val_loss / len(val_loader.dataset)
        val_losses.append(avg_val_loss)
       

        # 打印指标
        acc = accuracy_score(val_true, val_pred)
        print(f"Epoch [{epoch+1}/{num_epochs}] | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {acc:.4f}")

        scheduler.step()
        early_stop(avg_val_loss, model)
        if early_stop.early_stop:
            print("EarlyStopping")
            break

    # =============================
    # 测试集评价
    # =============================
    model.load_state_dict(torch.load(model2save + '.pth'))
    model.eval()
    test_true, test_pred = [], []
    test_logits = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device).long()
            logits = model(xb)
            y_prob = torch.softmax(logits, dim=1).cpu().numpy()
            preds = torch.argmax(logits, dim=1)
            test_pred.extend(preds.cpu().numpy())
            test_true.extend(yb.cpu().numpy())
            test_logits.extend(y_prob)

    test_true = np.array(test_true)
    test_pred = np.array(test_pred)
    test_logits = np.array(test_logits)

    # 计算多种指标
    acc = accuracy_score(test_true, test_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(test_true, test_pred, average='macro')
    cm = confusion_matrix(test_true, test_pred)
    
    auc = roc_auc_score(test_true, test_logits, multi_class='ovr')


    print("===== Test Set Metrics =====")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}, AUC: {auc:.4f}")
    print("Confusion Matrix:")
    print(cm)

    return train_losses, val_losses, test_true, test_pred, cm, test_logits



base_dir = "./datasets/test_result/stats/"
all_feature = np.load(base_dir + "all_feature_stats.npy", allow_pickle=True).astype(np.float32)
all_label = np.load(base_dir + "all_label_cls.npy", allow_pickle=True).astype(np.int64)

feature_dim = all_feature.shape[-1]

save_dir = './result/'
os.makedirs(save_dir, exist_ok=True)


all_test_true, all_test_pred, all_test_prob = [], [], []
last_test_true, last_test_pred = [], []
for i in range(0, 5):

	tr_index, val_index, te_index = train_val_test_split(kfold=5, fold=i, num=all_feature.shape[0], randomseed=555)

#	val_index = te_index
	train_feature, train_label = all_feature[tr_index], all_label[tr_index]
	val_feature, val_label = all_feature[val_index], all_label[val_index]
	test_feature, test_label = all_feature[te_index], all_label[te_index]

	# 标准化
	mean_feature = train_feature.mean(axis=(0,1))
	std_feature = train_feature.std(axis=(0,1))
    std_feature = np.maximum(std_feature, 1e-8)
	train_feature = (train_feature - mean_feature) / std_feature
	val_feature = (val_feature - mean_feature) / std_feature
	test_feature = (test_feature - mean_feature) / std_feature


	# 转Tensor
	train_dataset = TensorDataset(torch.tensor(train_feature), torch.tensor(train_label))
	val_dataset = TensorDataset(torch.tensor(val_feature), torch.tensor(val_label))
	test_dataset = TensorDataset(torch.tensor(test_feature), torch.tensor(test_label))

	train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
	val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
	test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

	classes, counts = np.unique(train_label, return_counts=True)
	print("Classes:", classes, "Counts:", counts)


	model2save = save_dir + 'model_class_' + str(i)
	result2save = save_dir + 'new_testresult_class_' + str(i)

	# =============================
	# 模型 & 优化器 & 损失函数
	# =============================
	device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
	model = LSTMAttentionClassifier(input_dim=feature_dim, hidden_dim=256, num_layers=4, num_classes=3, dropout=0.1).to(device)

	loss_fn = CombinedLoss(num_classes=3, alpha=[1., 1, 1.5], gamma=2.0, smoothing=0.1)
	optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
	scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.5)
	num_epochs = 1000

	# =============================
	# 训练 & 测试
	# =============================
	train_losses, val_losses, test_true, test_pred, cm, test_logits = run(model, train_loader, val_loader, test_loader,
		                                                optimizer, loss_fn, scheduler, num_epochs=num_epochs, device=device)
        
	# 保存测试结果
	df = pd.DataFrame({"id": te_index, "true": test_true, "pred": test_pred})
	df.to_csv(save_dir + "new_test_results_classification" + str(i) + ".csv", index=False)
	print("Test results saved to" + save_dir + "test_results_classification" + str(i) + ".csv")
	all_test_pred.append(test_pred)
	all_test_true.append(test_true)
	all_test_prob.append(test_logits)



all_test_true = np.concatenate(all_test_true)
all_test_pred = np.concatenate(all_test_pred)
all_test_prob = np.concatenate(all_test_prob)

# 计算多种指标
acc = accuracy_score(all_test_true, all_test_pred)
precision, recall, f1, _ = precision_recall_fscore_support(all_test_true, all_test_pred, average='macro')
cm = confusion_matrix(all_test_true, all_test_pred)

print("===== ALL Test Set Metrics =====")
print(f"Accuracy: {acc:.4f}")
print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-score: {f1:.4f}")
print("Confusion Matrix:")
print(cm)
# 类别数
n_classes = all_test_prob.shape[1]

# =========================
# One-hot
# =========================

y_true_bin = label_binarize(all_test_true, classes=np.arange(n_classes))

# =========================
# 每个类别 ROC
# =========================

fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(n_classes):

    fpr[i], tpr[i], _ = roc_curve(
        y_true_bin[:, i],
        all_test_prob[:, i]
    )

    roc_auc[i] = auc(fpr[i], tpr[i])
np.save(save_dir + "fpr.npy", fpr)
np.save(save_dir + "tpr.npy", tpr)
np.save(save_dir + "roc_auc.npy", roc_auc)
np.save(save_dir + "y_true_bin.npy", y_true_bin)
np.save(save_dir + "all_test_prob.npy", all_test_prob)


# =========================
# Macro AUC
# =========================

macro_auc = roc_auc_score(
    y_true_bin,
    all_test_prob,
    average='macro',
    multi_class='ovr'
)

# =========================
# Micro AUC
# =========================

micro_auc = roc_auc_score(
    y_true_bin,
    all_test_prob,
    average='micro',
    multi_class='ovr'
)

print("Macro AUC:", macro_auc)
print("Micro AUC:", micro_auc)

# =========================
# 画图
# =========================

plt.figure(figsize=(7,7))

for i in range(n_classes):

    plt.plot(
        fpr[i],
        tpr[i],
        label=f'Class {i} AUC = {roc_auc[i]:.3f}'
    )

# 对角线
plt.plot([0,1], [0,1], linestyle='--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")

plt.legend()
plt.grid(True)
plt.savefig(save_dir + "roc.png")
plt.show()

