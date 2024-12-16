# Customer Behavior Analysis and Machine Learning Modeling

## Project Overview

This project focuses on analyzing customer behavior using machine learning techniques. The dataset includes various attributes of customer transactions, such as purchase amount, frequency of purchases, and demographic details. The project performs clustering, regression, and classification to extract insights and predict customer behavior.

The goal is to utilize machine learning models to better understand customer segments, forecast purchase behavior, and classify subscription status. This project combines technical rigor with actionable insights to assist businesses in decision-making.

---

## Key Features

1. **Data Preprocessing**:
   - Handled missing data and duplicate values.
   - Encoded categorical variables using one-hot encoding and label encoding.
   - Scaled numerical features for consistency.

2. **Exploratory Data Analysis (EDA)**:
   - Visualized customer purchase trends using bar charts, histograms, and scatter plots.
   - Identified relationships between variables like purchase amount, frequency, and subscription status.

3. **Clustering (K-Means)**:
   - Implemented K-Means clustering to segment customers into distinct groups based on:
     - **Purchase Amount**  
     - **Frequency of Purchases**  
   - Visualized clusters to interpret customer segments.

4. **Regression Analysis**:
   - **Ridge Regression** was applied to predict purchase amounts based on other features.
   - Evaluated model performance using metrics such as R-squared and Mean Squared Error (MSE).

5. **Classification**:
   - Used **Logistic Regression** and **XGBoost Classifier** to predict subscription status (Yes/No).
   - Assessed performance using accuracy, precision, recall, and confusion matrices.

6. **Visualization**:
   - Visualized model outputs and clustering results using Matplotlib and Seaborn for actionable insights.

---

## Dataset Description

The dataset contains the following key columns:

- **Customer ID**: Unique identifier for each customer.
- **Age**: Age of the customer.
- **Gender**: Male/Female.
- **Item Purchased**: Product category purchased by the customer.
- **Purchase Amount (USD)**: The monetary value of purchases.
- **Frequency of Purchases**: Weekly, Monthly, Quarterly, etc.
- **Subscription Status**: Whether the customer has an active subscription (Yes/No).

---

## Code Highlights

### Preprocessing

```python
# Encoding categorical features
encoder = LabelEncoder()
dataset['Frequency of Purchases'] = encoder.fit_transform(dataset['Frequency of Purchases'])
```

### Clustering

```python
# K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)
clusters = kmeans.fit_predict(clustering_features_scaled)
```

### Regression

```python
# Ridge Regression
ridge_model = Ridge(alpha=1.0)
ridge_model.fit(X_regression, y_regression)
```

### Classification

```python
# XGBoost Classifier
xgb_classifier = XGBClassifier(random_state=42)
xgb_classifier.fit(X_classification, y_classification)
```

---

## Results and Insights

1. **Clustering**:
   - Identified distinct customer segments based on spending behavior and purchase frequency.
   - Helpful in targeted marketing and personalized customer outreach.

2. **Regression**:
   - Ridge Regression demonstrated a good ability to predict purchase amounts with high accuracy.

3. **Classification**:
   - XGBoost Classifier achieved high precision and recall in predicting subscription status, showing its effectiveness for binary classification tasks.

---

## Technologies Used

- **Python Libraries**:
  - Data Handling: Pandas, NumPy
  - Visualization: Matplotlib, Seaborn
  - Machine Learning: Scikit-learn, XGBoost
- **Development Tools**:
  - Jupyter Notebook
  - GitHub for version control

---

## How to Use

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/Customer-Behavior-Analysis.git
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```
3. Run the Jupyter Notebook `Final.ipynb` to reproduce the analysis and results.

---

## Future Enhancements

- Incorporate advanced techniques like PCA for dimensionality reduction.
- Explore other models such as Random Forest for regression and classification.
- Automate hyperparameter tuning using GridSearchCV or Optuna.

---

Feel free to customize this README further based on your GitHub username or any additional details you want to highlight! Let me know if you'd like help refining it further.
