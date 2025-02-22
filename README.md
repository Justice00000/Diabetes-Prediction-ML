# Diabetics Prediction

## Problem Statement
This project aims to develop a machine-learning model to predict the likelihood of diabetes in patients based on various medical features. The dataset used is publicly available on Kaggle via https://www.kaggle.com/datasets/ishandutta/early-stage-diabetes-risk-prediction-dataset and is aligned with healthcare diagnostics.

## Dataset
The dataset contains multiple features relevant to diabetes prediction, including glucose levels, blood pressure, BMI, and insulin levels. It has been preprocessed for model training and evaluation.

## Implemented Models
- **Logistic Regression**
- **Support Vector Machine (SVM)**
- **Basic Neural Network (No optimizations)**
- **Optimized Neural Network with Regularization and Hyperparameter Tuning**

## Training Results

| Instance | Optimizer | Regularizer | Epochs | Early Stopping | Layers | Learning Rate | Accuracy | Precision | Recall | F1 Score |
|----------|------------|------------|--------|----------------|--------|---------------|----------|-----------|--------|----------|
| 1        | Adam       | None       | 10     | No             | 2      | 0.001         | 0.9231   | 0.9315    | 0.9577 | 0.9097   |
| 2        | RMSprop    | L1         | 20     | Yes            | 3      | 0.0005        | 0.9904   | 0.9861    | 1.0000 | 0.9888   |
| 3        | Adam       | L2         | 30     | Yes            | 4      | 0.0001        | N/A      | N/A       | N/A    | N/A      |
| 4        | Adam       | L1 & L2    | 40     | Yes            | 5      | 0.00005       | N/A      | N/A       | N/A    | N/A      |
| 5        | RMSprop    | None       | 50     | No             | 6      | 0.00001       | N/A      | N/A       | N/A    | N/A      |

## Summary of Findings
- The optimized neural network with multiple layers and tuned hyperparameters performed better than the basic model.
- Regularization (L1/L2) improved generalization, preventing overfitting.
- The ML algorithm (Logistic Regression) performed well but was outperformed by the optimized neural network.

## Running the Notebook
1. Clone this repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Open and run `notebook.ipynb`.
4. The best-trained model can be loaded from the `saved_models` directory.

## Video Presentation
