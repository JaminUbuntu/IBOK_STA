
# 📊 Predicting Customer Churn Using Traditional Machine Learning Models: A Comparative Analysis

## 🔍 Overview

Customer Churn is a pivotal business metric that influences strategic decisions across industries. In the age of Data-Driven Enterprises, proactively identifying customers at risk of leaving is vital for sustaining profitability and enhancing customer retention. This project explores the efficacy of Traditional Machine Learning Models in predicting churn in the Telecommunications sector.

By leveraging Real-World Customer Behavior Data, we conduct a Comparative Analysis of various Classical Machine Learning Algorithms, optimizing them for performance and interpretability. The insights drawn from this study are not only academically robust but are also directly transferable to Real-World Business Applications.

---

## 🧠 Project Objectives

- Evaluate and compare Traditional Machine Learning Algorithms for Churn Prediction.
- Build a Reproducible and Scalable Pipeline for Data Preprocessing, Modeling, and Evaluation.
- Identify the most influential Features driving Customer Attrition.
- Assess the practical implications of Model Deployment in Real-World Telecom Operations.

---

## 🛠️ Technologies Used

- **Programming Language:** Python
- **Data Manipulation & Visualization:** Pandas, NumPy, Matplotlib, Seaborn
- **Machine Learning Models:** 
  - Logistic Regression
  - Random Forest
  - Support Vector Machines (SVM)
  - K-Nearest Neighbors (KNN)
  - Gaussian Naive Bayes
  - XGBoost
- **Model Evaluation & Selection:** Scikit-Learn, GridSearchCV, Recursive Feature Elimination (RFE)
- **Imbalance Handling:** SMOTE (Synthetic Minority Oversampling Technique), Random Under Sampler (RUS)
- **Pipeline Management:** Sklearn.Pipeline, Imbalanced-Learn Pipeline

---

## 📁 Dataset

The Dataset used for this project is sourced from a publicly available **Telecommunications Customer Churn** Dataset on [Kaggle](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets). It comprises 2,666 Records and 20 Features, capturing detailed User Activity, Charges, and Service Engagement Metrics.

---

## 🔬 Methodology

1. **Data Cleaning & EDA:** Ensured quality by checking for Null Values, Duplicates, and Outliers. Performed Exploratory Data Analysis to understand Feature Relationships.
2. **Feature Engineering:** Encoded Categorical Variables and Normalized Numerical Features for uniform Model Training.
3. **Handling Class Imbalance:** Employed both Oversampling (SMOTE) and Undersampling (RUS) to manage Minority Class Representation.
4. **Model Implementation:** Deployed Models in a Unified Pipeline enabling comparison under consistent preprocessing steps.
5. **Performance Metrics:** Models evaluated using Accuracy, Precision, Recall, F1-Score, ROC-AUC, and Logarithmic Loss.
6. **Model Interpretability:** Applied Feature Importance Analysis using Random Forest and Recursive Feature Elimination (RFE).

---

## 📈 Key Insights & Results

- **XGBoost** emerged as the top performer with **95% Accuracy**, highlighting its superior learning ability through Boosting Mechanisms.
- **Random Forest** closely followed, offering Interpretability via Feature Importance.
- Handling Class Imbalance significantly improved Recall and overall robustness.
- Feature Reduction via RFE enhanced Model Generalizability while preserving performance.

---

## 🧩 Real-World Applications

This project exemplifies how Traditional Machine Learning can be leveraged for **Predictive Analytics in Customer Relationship Management**, specifically in:
- **Telecommunication:** Proactively identifying at-risk Customers to reduce Churn.
- **Subscription-Based Services:** Tailoring Retention Strategies based on Predictive Insights.
- **Digital Marketing:** Designing Targeted Campaigns using Churn Probability Scores.

---

## 🔗 How to Leverage This Repository

1. Clone the repo:
   ```bash
   git clone https://github.com/JaminUbuntu/IBOK_ML-CW.git
   ```
2. Explore the core Jupyter Notebook: [`ML_Coursework_Teleco_Churn_Project.ipynb`](https://github.com/JaminUbuntu/IBOK_ML-CW/blob/main/ML_Coursework_Teleco_Churn_Project.ipynb)
3. Examine how Pipelines are structured for Multi-Model Evaluation.
4. Use the Notebook as a blueprint for similar Classification Tasks in Business or Research.

---

## 🧭 Future Directions

- Introduce **Explainable AI** (e.g., SHAP, LIME) for Model Transparency.
- Integrate **Profit-Based Metrics** for Decision-Making aligned with ROI.
- Expand the analysis to include **Deep Learning Models** and **Time-Based Churn Forecasting**.

---

## 📚 Citation

If you use this work for Research or Academic Purposes, please cite:
> Ibok, B. (2024). *Predicting Customer Churn Using Traditional Machine Learning Models: A Comparative Analysis*. Coventry University.

---

## 🎓 Academic Context

This project is part of the Coursework for **Module 7072CEM – Machine Learning** at Coventry University. It satisfies key Learning Outcomes including:
- Algorithm Implementation and Evaluation
- Solving Real-World Classification Problems
- Ethical and Professional Considerations in AI Systems

---

## 📬 Contact

**Author:** Benjamin Ibok  
**Institution:** Coventry University  
**Email:** ibokb@coventry.ac.uk  
**Personal Email:** benjaminsibok@gmail.com

---

## ⚙️ Environment Setup

This project was originally developed using **Google Colab**. However, the code is structured to be portable and will also be deployed as a **Kaggle Notebook** for broader accessibility.

### Installation Requirements

To run this notebook locally or in a cloud-based environment:

```bash
pip install -r requirements.txt
```

#### `requirements.txt` includes:
- Python >= 3.8
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- xgboost
- imbalanced-learn

---

## 📊 Visualizations & Model Evaluation

Several insights were derived through comprehensive visualizations, including:

- **Churn Distribution**
- **Feature Importance Analysis** (via Random Forest)
- **Boxplots** to identify Outliers
- **Pairplots** to visualize Feature Correlations
- **Confusion Matrices** before and after SMOTE and Feature Selection
- **ROC Curves** to compare Model Performances

![Churn Distribution](/churn_distribution.png)
![Feature Importance](/feature_importance.png)
![Model ROC Comparison](outputs/roc_comparison.png)

---

## 🤝 Contribution Guidelines

We welcome contributions! To contribute:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature-name`).
3. Commit your changes (`git commit -m 'Add feature'`).
4. Push to the branch (`git push origin feature-name`).
5. Open a Pull Request.

Please adhere to PEP8 standards and include detailed docstrings.

---

## 💾 Model Inference

To reuse the trained models without retraining:

1. Load the model using `joblib`:
```python
from joblib import load
model = load("models/logistic_model.joblib")
```

2. Predict:
```python
prediction = model.predict(new_data)
```

Trained model files will be added under the `/models` directory.

---

## 🏷️ Project Badges

![Python](https://img.shields.io/badge/python-3.9-blue.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Platform: Google Colab](https://img.shields.io/badge/platform-Colab-green.svg)
![Status: Active](https://img.shields.io/badge/status-active-brightgreen)

---

## ❓ FAQ / Known Issues

- **Q: Where can I find the dataset?**  
  A: The dataset is available on [Kaggle](https://www.kaggle.com/datasets/mnassrib/telecom-churn-datasets).

- **Q: I'm seeing issues with imbalanced-learn installation.**  
  A: Ensure you are using Python >= 3.8 and try:  
  `pip install imbalanced-learn`

- **Q: Will the notebook run on Kaggle?**  
  A: Yes, we are working on a Kaggle-compatible version. Ensure the dataset is uploaded or linked correctly in Kaggle.

---
