# IORGAI-PRED
# IOrgAI: AI-Driven Immune Organoid Analysis

This repository contains the code for **IOrgAI (Immune Organoid Artificial Intelligence)**, an AI-driven framework designed for automated immune organoid image analysis and the prediction of vaccine-induced immune response intensities.

IOrgAI consists of two main modules:

- **IOrgAI-Seg (Instance Segmentation):** Identifies and segments individual immune organoids from microscopy images, enabling quantitative analysis of organoid phenotypes. The implementation of IOrgAI-Seg is available in a separate repository: [IOrgAI-Seg](https://github.com/HGCSDN/Organoid_Overlap_Instance_Segmentation/tree/main/OrgAI).

- **IOrgAI-Pred (Temporal Prediction):** Uses phenotypic features extracted from IOrgAI-Seg segmentation results at three time points (24, 36, and 48 h) to predict vaccine-induced immune response intensities.

## Workflow

The overall workflow consists of three steps:

1. **Instance Segmentation:** Microscopy images are processed using IOrgAI-Seg to identify and segment individual immune organoids.
2. **Feature Extraction:** Quantitative phenotypic features are extracted from the segmentation results and organized into temporal feature sequences.
3. **Prediction:** The extracted temporal features are used to train and evaluate IOrgAI-Pred for two independent tasks:
   - **Regression:** Predicting continuous immune response intensities.
   - **Classification:** Categorizing immune response intensities into three levels: Low, Medium, and High.
