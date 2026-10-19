from scipy import stats
import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold


def train_val_test_split(kfold=10, fold=0, num=0, randomseed=123):
    id = list(range(num))

    import random
    random.seed(randomseed)
    random.shuffle(id)

    kf = KFold(n_splits=kfold, random_state=randomseed, shuffle=True)
    kf2 = KFold(n_splits=kfold - 1, shuffle=True, random_state=randomseed) #999

    test_index = list()
    train_index = list()
    val_index = list()

    for tr, te in kf.split(np.array(id)):
        test_index.append(te)
        tr_id, val_id = list(kf2.split(tr))[0]
        train_index.append(tr[tr_id])
        val_index.append(tr[val_id])

    train_id = train_index[fold]
    test_id = test_index[fold]
    val_id = val_index[fold]

    return train_id, val_id, test_id
