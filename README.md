# 🧠 Parkinson's Disease Detection using Machine Learning

This repository contains a machine learning pipeline to detect **Parkinson's Disease** using a range of classification algorithms. The goal is to predict the presence of Parkinson’s disease based on vocal measurements.

## 📊 Project Overview

This project involves:

- **Data preprocessing** using `StandardScaler` and `Label Encoding`
- **Balancing the dataset** using `SMOTE` (Synthetic Minority Over-sampling Technique)
- **Model building** with multiple ML algorithms
- **Hyperparameter Tuning** using `GridSearchCV`
- **Model evaluation** with metrics such as Accuracy, F1 Score, Precision, and Recall

## 🚀 Tools & Libraries Used

- Python
- Pandas, NumPy
- Scikit-learn
- imbalanced-learn (`SMOTE`)
- Matplotlib, Seaborn (for visualization)

## 🧪 Dataset

The dataset used in this project is related to voice features of individuals. Each instance contains multiple biomedical voice measurements from people, some of whom have Parkinson’s disease.

📁 File: `Parkinson_Disease_Detection.ipynb`

## 🔄 Preprocessing Steps

1. **Standardization** using `StandardScaler`
2. **Handling Class Imbalance** using `SMOTE`

## 🧠 Models Trained & Performance

| ML Model           | Accuracy | F1 Score | Recall  | Precision |
|--------------------|----------|----------|---------|-----------|
| SVC (Tuned)        | 1.0000   | 1.0000   | 1.0000  | 1.0000    |
| Random Forest      | 0.9661   | 0.9667   | 0.9355  | 1.0000    |
| Decision Tree      | 0.9322   | 0.9310   | 0.9310  | 0.9310    |
| Logistic Regression| 0.8644   | 0.8667   | 0.8387  | 0.8966    |

✅ **SVC (Support Vector Classifier)** with hyperparameter tuning using **GridSearchCV** gave **perfect performance** on the test set.

# 📈 Evaluation Metrics
Accuracy: Overall correctness

Precision: How many selected items are relevant

Recall: How many relevant items are selected

F1 Score: Harmonic mean of Precision and Recall

# 📌 Key Takeaways
Data balancing significantly helped in improving recall and F1 score

SVC with GridSearchCV yielded the best results

The pipeline is easily extendable to other biomedical classification problems

## 📄 License

This project is licensed under the MIT License.

---
**Made ❤️ by Brijesh Rakhasiya**
