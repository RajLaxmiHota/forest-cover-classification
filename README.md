# Forest Cover Type Classification

## a. Problem Statement

The objective of this assignment is to implement multiple machine learning classification models to predict the **Forest Cover Type** based on cartographic and environmental features.

This project includes:
- Implementation of six classification models
- Evaluation using multiple performance metrics
- Model comparison
- Deployment using Streamlit

This is a multi-class classification problem with 7 classes.

---

## b. Dataset Description

- **Dataset Name:** Covertype Dataset (UCI Machine Learning Repository)
- **Total Instances:** 581,012
- **Total Features:** 54
- **Target Variable:** Cover_Type
- **Number of Classes:** 7

The dataset includes environmental and cartographic attributes such as elevation, slope, soil type, and wilderness area indicators used to classify forest cover type.

---

## c. Models Used

1. Logistic Regression  
2. Decision Tree Classifier  
3. K-Nearest Neighbors (KNN)  
4. Naive Bayes (Gaussian)  
5. Random Forest (Ensemble)  
6. XGBoost (Ensemble Boosting)  

---

## Model Comparison Table

| ML Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|----------|----------|------|-----------|--------|----------|------|
| Logistic Regression | 0.7235 | 0.9363 | 0.7109 | 0.7235 | 0.7138 | 0.5469 |
| Decision Tree | 0.9389 | 0.9452 | 0.9389 | 0.9389 | 0.9389 | 0.9019 |
| KNN | 0.9284 | 0.9839 | 0.9282 | 0.9284 | 0.9282 | 0.8849 |
| Naive Bayes | 0.0895 | 0.7970 | 0.4940 | 0.0895 | 0.0577 | 0.0688 |
| Random Forest | 0.9533 | 0.9978 | 0.9534 | 0.9533 | 0.9530 | 0.9248 |
| XGBoost | 0.8682 | 0.9862 | 0.8683 | 0.8682 | 0.8674 | 0.7871 |

---

## Observations

| ML Model | Observation |
|----------|-------------|
| Logistic Regression | Moderate performance. Struggles with complex non-linear patterns. |
| Decision Tree | Strong performance. Captures non-linear relationships effectively but may overfit slightly. |
| KNN | Performs well after scaling. Distance-based learning suits this dataset. |
| Naive Bayes | Very poor performance due to unrealistic feature independence assumptions. |
| Random Forest | Best performing model overall with highest accuracy and MCC. |
| XGBoost | Strong AUC performance but slightly lower accuracy than Random Forest. |

---

## Streamlit Application Features

- CSV dataset upload option  
- Model selection dropdown  
- Display of evaluation metrics  
- Confusion matrix visualization  
- Classification report display  

---

## Technologies Used

Python, Scikit-learn, XGBoost, Pandas, NumPy, Seaborn, Matplotlib, Streamlit
