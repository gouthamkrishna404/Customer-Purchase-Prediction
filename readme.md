# Customer Purchase Prediction

## Project Overview
This project demonstrates the development of a machine learning classification system designed to predict whether a customer is likely to make a purchase. Using demographic and behavioral data, the system identifies patterns in customer behavior and evaluates multiple classification algorithms to select the most effective model.

The primary goal is to provide actionable insights into customer purchase behavior and build a predictive model that can aid in targeted marketing strategies.

---

## Dataset
The dataset used in this project is available on Kaggle: [Customer Purchase Behavior Dataset]([https://www.kaggle.com/datasets/your-dataset-link](https://www.kaggle.com/datasets/rabieelkharoua/predict-customer-purchase-behavior-dataset))  

This dataset contains anonymized customer data and includes demographic and behavioral features for purchase prediction.

---

**Features include:**
- `Age` – Age of the customer  
- `Gender` – Gender of the customer  
- `Annual Income` – Yearly income of the customer  
- `Number of Purchases` – Total purchases made  
- `Product Category` – Category of products purchased  
- `Time Spent on Website` – Average time spent browsing  
- `Loyalty Program` – Whether the customer is enrolled  
- `Discounts Availed` – Discounts the customer has used  

**Target Variable:**
- `PurchaseStatus` (0 = No Purchase, 1 = Purchase)

---

## Machine Learning Workflow

### 1. Data Loading & Exploration
- Loaded the dataset using **pandas**  
- Examined the dataset’s shape, column types, and missing values  
- Identified patterns and distributions  

### 2. Exploratory Data Analysis (EDA)
- Visualized the distribution of purchase vs no purchase
- ![Purchase Distribution](images/purchase_distribution.png)
- Created correlation heatmaps to identify feature relationships
- ![Feature Correlation](images/feature_correlation.png)
- Explored feature-to-feature interactions  

### 3. Data Preprocessing
- Handled missing values and encoded categorical features  
- Scaled numeric features using `StandardScaler`  
- Split dataset into training and testing sets (80/20)  

### 4. Model Building
Implemented multiple classification models and compared performance:  
- Logistic Regression  
- Decision Tree  
- Random Forest (with hyperparameter tuning using GridSearchCV)  
- K-Nearest Neighbors (KNN)  

### 5. Model Evaluation
- Evaluated models using the following metrics:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1-Score  
  - Confusion Matrix  

### 6. Feature Importance
- Identified key predictors using **Random Forest feature importance**  
- Visualized contributions of each feature to model predictions  

---

## Results
- **Random Forest** emerged as the best-performing model based on accuracy and other evaluation metrics.  
- Most impactful features for predicting purchase behavior:  
  - `Annual Income`  
  - `Time Spent on Website`  
  - `Number of Purchases`  

These insights can guide business decisions for targeted promotions and customer engagement strategies.

---

## Technologies & Libraries
- **Programming Language:** Python  
- **Data Manipulation:** pandas, NumPy  
- **Visualization:** Matplotlib, Seaborn  
- **Machine Learning:** Scikit-learn  
- **Model Persistence:** joblib  
- **Deployment:** Streamlit  

---
## How to Access
Visit the live Streamlit app at: [Customer-Purchase-Prediction]([https://share.streamlit.io/your-username/your-repo/main](https://customer-purchase-prediction-404.streamlit.app))

