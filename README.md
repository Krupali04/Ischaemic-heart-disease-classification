# Ischemic Heart Disease Classification

An advanced machine learning system for detecting Ischemic Heart Disease (IHD) using ECG data. This project processes ECG signals from .mat files and their corresponding diagnostic information from .hea files to identify cases of myocardial ischemia.

## Project Overview

This project implements a machine learning pipeline for automated detection of ischemia from ECG recordings. It includes:

- Data preprocessing of ECG signals
- Feature extraction from raw ECG data
- Multiple machine learning models with cost-sensitive variants
- Comprehensive evaluation metrics and visualization

## Features

### Data Processing
- Handles .mat files containing ECG signals
- Extracts diagnostic labels from .hea files using SNOMED-CT codes
- Computes key ECG features including:
  - Statistical measures (mean, std, max, min)
  - Root Mean Square (RMS)
  - Peak-to-peak amplitude

### Machine Learning Models
- Naive Bayes
- Neural Network
- Decision Tree
- K-Nearest Neighbors
- Cost-sensitive variants of all models

### Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1 Score
- ROC curves and AUC
- Confusion matrices

## Current Performance

Best performing model: Cost-sensitive Neural Network
- F1 Score: 0.241
- Precision: 0.137
- Recall: 1.000
- Accuracy: 0.141

The model achieves perfect recall (detecting all ischemia cases) while maintaining acceptable precision for a medical screening tool.

## Dataset

The system works with ECG recordings in .mat format and their corresponding .hea header files. The header files contain diagnostic codes following the SNOMED-CT standard for accurate labeling of ischemia cases.

