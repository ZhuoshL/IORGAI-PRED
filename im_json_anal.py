import os
import numpy as np
import pandas as pd
import cv2
from pycocotools.coco import COCO
from pycocotools import mask as maskUtils

# ===== 路径 =====
base_dir = './'
im_dir = os.path.join(base_dir, 'datasets/test/')
json_file = os.path.join(base_dir, 'datasets/test_result/inference.json')
sa_dir = os.path.join(base_dir, 'datasets/test_result/stats')
os.makedirs(sa_dir, exist_ok=True)

category_ids = [1, 2, 3, 4]
PIXEL_AREA = 25
BORDER_MARGIN = 3  


def get_mask_from_ann(ann, height, width):
    segm = ann['segmentation']
    if isinstance(segm, list):
        rles = maskUtils.frPyObjects(segm, height, width)
        rle = maskUtils.merge(rles)
    elif isinstance(segm, dict):
        if isinstance(segm.get('counts'), list):
            rle = maskUtils.frPyObjects(segm, height, width)
        else:
            rle = segm
    else:
        raise TypeError(f"Unsupported segmentation type: {type(segm)}")
    mask = maskUtils.decode(rle)
    if mask.ndim == 3:
        mask = np.any(mask > 0, axis=2)
    return mask > 0


def extract_all_features(anns, image, height, width):
    """提取全部满足原边界条件的实例，返回逐实例 (annotation_id, area, gray)。"""
    instances = []
    for ann in anns:
        x, y, w, h = map(int, ann['bbox'])
        if not (x > BORDER_MARGIN and y > BORDER_MARGIN
                and width - x - w > BORDER_MARGIN
                and height - y - h > BORDER_MARGIN):
            continue

        mask = get_mask_from_ann(ann, height, width)
        pixel_count = int(np.count_nonzero(mask))
        if pixel_count == 0:
            continue
        area = float(pixel_count * PIXEL_AREA)
        gray = float(np.mean(image[mask]))
        instances.append((ann.get('id'), area, gray))
    return instances


def summarize(instances):
    if not instances:
        return [0.0] * 6
    areas = np.array([v[1] for v in instances], dtype=np.float64)
    grays = np.array([v[2] for v in instances], dtype=np.float64)
    return [float(np.mean(areas)), float(np.std(areas)), float(np.median(areas)),
            float(np.mean(grays)), float(np.std(grays)), float(np.median(grays))]


def main():
    coco = COCO(json_file)
    instance_rows, stats_rows, img_names = [], [], []
    stats_columns = ['area_mean', 'area_std', 'area_median',
                     'gray_mean', 'gray_std', 'gray_median']

    for img_id in coco.getImgIds():
        img_info = coco.loadImgs(ids=img_id)[0]
        img_name = img_info['file_name']
        image_path = os.path.join(im_dir, img_name)
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f'Cannot read image: {image_path}')
        height, width = image.shape[:2]
        ann_ids = coco.getAnnIds(imgIds=img_id, catIds=category_ids)
        anns = coco.loadAnns(ids=ann_ids)
        instances = extract_all_features(anns, image, height, width)

        stats_rows.append({
            'img_name': img_name, 'image_id': img_id,
            'instance_count': len(instances),
            **dict(zip(stats_columns, summarize(instances)))
        })
        img_names.append(img_name)


    df_stats = pd.DataFrame(stats_rows, columns=[
        'img_name', 'image_id', 'instance_count', *stats_columns
    ])

    features_stats = df_stats[stats_columns].to_numpy(dtype=np.float32)
    df_stats.to_csv(os.path.join(sa_dir, 'features_stats_all.csv'), index=False)
    np.save(os.path.join(sa_dir, 'features_stats_all.npy'), features_stats)
    pd.DataFrame({'img_name': img_names}).to_csv(
        os.path.join(sa_dir, 'img_names.csv'), index=False
    )

    print(f'Images: {len(img_names)}')
    print(f'Stats shape: {features_stats.shape}')
    print(f'Output directory: {sa_dir}')


if __name__ == '__main__':
    main()
