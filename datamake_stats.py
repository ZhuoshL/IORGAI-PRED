import os
import numpy as np
import pandas as pd

# Paths
base_path = './datasets/test_result/stats/'
feature_path = os.path.join(base_path, 'features_stats_all.npy')
img_path = os.path.join(base_path, 'img_names.csv')
# One Excel file with three columns: id, label_reg, label_cls.
label_path = os.path.join(base_path, 'labels.xlsx')
ID_COLUMN = 'id'
REG_COLUMN = 'label_reg'
CLS_COLUMN = 'label_cls'
label_label = {'High': 2, 'Medium': 1, 'Low': 0}
times = ('24', '36', '48')

# Image-level features: area mean/std/median, gray mean/std/median.
img_feature = np.load(feature_path, allow_pickle=False)
img_series = pd.read_csv(img_path)['img_name'].astype(str)
if img_feature.ndim != 2 or img_feature.shape[1] != 6:
    raise ValueError(f'Expected features (N, 6), got {img_feature.shape}')
if len(img_feature) != len(img_series):
    raise ValueError('Feature rows and image names do not match')
if img_series.duplicated().any():
    raise ValueError('Duplicate image names in img_names.csv')
if not np.isfinite(img_feature).all():
    raise ValueError('Feature array contains NaN or infinity')
image_lookup = dict(zip(img_series, img_feature))

# Read regression and classification labels from the same row of one Excel file.
label_df = pd.read_excel(label_path)
required = [ID_COLUMN, REG_COLUMN, CLS_COLUMN]
missing_columns = [c for c in required if c not in label_df.columns]
if missing_columns:
    raise ValueError(f'Missing Excel columns: {missing_columns}; found: {list(label_df.columns)}')

label_df = label_df[required].copy()
if label_df.isna().any().any():
    raise ValueError('Excel contains missing IDs or regression/classification labels')

raw_ids = pd.to_numeric(label_df[ID_COLUMN], errors='raise')
if not np.isfinite(raw_ids).all() or not (raw_ids == raw_ids.astype(np.int64)).all():
    raise ValueError('Sample IDs must be finite integers')
label_df[ID_COLUMN] = raw_ids.astype(np.int64)
if label_df[ID_COLUMN].duplicated().any():
    raise ValueError('Duplicate sample IDs in Excel')

label_df[REG_COLUMN] = pd.to_numeric(label_df[REG_COLUMN], errors='raise')
if not np.isfinite(label_df[REG_COLUMN].to_numpy(dtype=float)).all():
    raise ValueError('Regression labels contain NaN or infinity')

levels = label_df[CLS_COLUMN].astype(str).str.strip().str.title()
unknown = sorted(set(levels) - set(label_label))
if unknown:
    raise ValueError(f'Unknown classification labels: {unknown}')
label_df[CLS_COLUMN] = levels.map(label_label).astype(np.int64)

label_lookup = {
    int(row[ID_COLUMN]): (float(row[REG_COLUMN]), int(row[CLS_COLUMN]))
    for _, row in label_df.iterrows()
}

# Match three time points and both label types for each sample.
ids = list(dict.fromkeys(int(name.split('-')[0]) for name in img_series))
X, Y_reg, Y_cls, save_ids = [], [], [], []
for sample_id in ids:
    if sample_id not in label_lookup:
        print(f'Skip {sample_id}: missing label')
        continue
    names = [f'{sample_id}-{t}.jpg' for t in times]
    missing = [name for name in names if name not in image_lookup]
    if missing:
        print(f'Skip {sample_id}: missing images {missing}')
        continue
    X.append([image_lookup[name] for name in names])
    reg_value, cls_value = label_lookup[sample_id]
    Y_reg.append(reg_value)
    Y_cls.append(cls_value)
    save_ids.append(sample_id)

if not X:
    raise ValueError('No samples have all three images and both labels')
X_array = np.asarray(X, dtype=np.float32)
Y_reg_array = np.asarray(Y_reg, dtype=np.float32)
Y_cls_array = np.asarray(Y_cls, dtype=np.int64)
save_ids = np.asarray(save_ids, dtype=np.int64)

# Keep raw features; training-fold normalization belongs in the training script.
np.save(os.path.join(base_path, 'all_feature_stats.npy'), X_array)
np.save(os.path.join(base_path, 'all_label.npy'), Y_reg_array)
np.save(os.path.join(base_path, 'all_label_cls.npy'), Y_cls_array)

feature_names = ('area_mean', 'area_std', 'area_median', 'gray_mean', 'gray_std', 'gray_median')
columns = [f'{name}_{t}' for t in times for name in feature_names]
df = pd.DataFrame(X_array.reshape(len(X_array), -1), columns=columns)
df.insert(0, 'id', save_ids)
df['label_reg'] = Y_reg_array
df['label_cls'] = Y_cls_array
df.to_excel(os.path.join(base_path, 'all_feature_stats.xlsx'), index=False)
print('Features:', X_array.shape)
print('Regression labels:', Y_reg_array.shape)
print('Classification labels:', Y_cls_array.shape)
print('Class counts:', dict(zip(*np.unique(Y_cls_array, return_counts=True))))
print('Saved to:', base_path)
