# Customer Churn Prediction Using Machine Learning

## 📌 Project Overview

This project focuses on developing a machine learning model to predict customer churn using Python and Scikit-learn. Customer churn refers to customers discontinuing their use of a company's products or services.

The objective is to identify customers who are likely to churn and help businesses develop effective customer retention strategies.


## 🎯 Objectives

* Prepare and clean customer data for machine learning.
* Perform exploratory data analysis (EDA).
* Build and compare multiple classification models.
* Evaluate model performance using appropriate metrics.
* Identify customers with a high risk of churn.
* Recommended action to prevent the churn
  

## 📂 Project Structure

```text
customer-churn-prediction/
│
├── churn_model(1).ipynb
├── Machine_Learning_Churn_Report.docx
├── README.md
└── requirements.txt
```


## 📊 Dataset Description

The project uses a customer dataset containing **10,000 records and 16 original columns**.

The dataset includes customer information such as:

* Customer profile and demographics
* Contract type and tenure
* Service usage and billing details
* Customer satisfaction
* Complaints and support calls
* Churn status

The target variable is `Churn`:

* `0` — Customer retained
* `1` — Customer churned

The dataset contains 7,740 retained customers and 2,260 churned customers, indicating class imbalance.

## 🛠️ Technologies Used

* **Python** — Programming language
* **Pandas** — Data manipulation and analysis
* **NumPy** — Numerical computations
* **Matplotlib** — Data visualization
* **Seaborn** — Exploratory data visualization
* **Scikit-learn** — Machine learning model development and evaluation
* **VS Code** — Interactive development environment

## 🔍 Project Workflow

### 1. Data Cleaning and Preprocessing

* Checked data types and missing values.
* Checked for duplicate records and customer IDs.
* Imputed missing categorical values using the mode.
* Imputed missing numerical values using the median.
* Removed `Customer_ID` from the model features.

### 2. Exploratory Data Analysis

Explored customer churn patterns and examined relationships between churn and variables such as contract type, tenure, customer satisfaction, complaints, and support calls.

Visualizations were used to understand the target distribution and potential churn indicators.

### 3. Feature Engineering and Preparation

* Separated input features from the target variable.
* Identified numerical and categorical columns.
* Applied `StandardScaler` to numerical features.
* Applied `OneHotEncoder` to categorical features.
* Used a `ColumnTransformer` and Scikit-learn pipelines to combine preprocessing and modeling.

### 4. Model Training

The following machine learning algorithms were compared:

1. Logistic Regression
2. Balanced Logistic Regression
3. Random Forest Classifier

Balanced Logistic Regression uses `class_weight="balanced"` to account for the unequal distribution of churn and non-churn customers.

### 5. Model Evaluation

The models were evaluated using:

* **Accuracy:** Overall proportion of correct predictions.
* **Precision:** Proportion of predicted churners who actually churned.
* **Recall:** Proportion of actual churners correctly identified.
* **F1-score:** Harmonic mean of precision and recall.
* **ROC-AUC:** Ability to distinguish between churn and non-churn customers.

A confusion matrix was also used to analyze correct predictions and classification errors.

## 📈 Model Performance

The notebook reports the following approximate performance for the three compared models:

| Model                        | Accuracy | Precision | Recall | F1-score | ROC-AUC |
| ---------------------------- | -------: | --------: | -----: | -------: | ------: |
| Logistic Regression          |    80.0% |     63.0% |  33.0% |    43.0% |   79.0% |
| Balanced Logistic Regression |    76.6% |     48.6% |  60.2% |    53.8% |   78.8% |
| Random Forest                |    79.0% |     59.0% |  24.0% |    34.0% |   77.0% |

*Values are rounded from the notebook's model comparison. The later final pipeline reports approximately 77% accuracy, 49% precision, 60% recall, 54% F1-score, and 79% ROC-AUC.*

### Best-Performing Approach

Balanced Logistic Regression is the preferred baseline when identifying as many potential churners as possible is important.

Although its accuracy is slightly lower than standard Logistic Regression, it achieves higher recall and a better F1-score, making it more useful for prioritizing customer retention efforts.

## 📉 Visualizations

The project includes visual analysis to support model evaluation, including:

* Churn distribution and exploratory analysis plots
* Model performance comparison
* Confusion matrix
* Customer churn-risk segmentation

These visualizations help explain model strengths, weaknesses, and classification errors.

## ⚠️ Limitations

* The dataset has an imbalanced target variable.
* False positives may lead to unnecessary retention campaigns.
* False negatives mean some customers likely to churn will not be identified.
* Simple imputation may not fully capture patterns in missing data.
* A single train/test split may not represent performance on future data.
* Model predictions indicate associations, not necessarily causes of churn.

## 🚀 Future Improvements

* Use stratified cross-validation for more reliable evaluation.
* Improve preprocessing by fitting imputation steps within the training pipeline.
* Evaluate precision-recall curves and PR-AUC.
* Tune model hyperparameters.
* Compare additional algorithms, including gradient boosting.
* Optimize the classification threshold based on business costs.
* Monitor model performance and data drift after deployment.


## 📌 Conclusion

This project demonstrates the end-to-end machine learning workflow for customer churn prediction, including data cleaning, exploratory analysis, preprocessing, model training, evaluation, and risk segmentation.

Balanced Logistic Regression provides a useful baseline for identifying potential churners. Further improvements in validation, feature engineering, threshold selection, and monitoring could increase its practical value.


---

*This project is intended for educational purposes and demonstrates fundamental machine learning techniques using Python and Scikit-learn.*
