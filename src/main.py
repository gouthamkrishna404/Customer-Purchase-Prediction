import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# STEP 1: LOAD DATA

df = pd.read_csv("data/customer_purchase_data.csv")

print("\nFirst 5 rows:")
print(df.head())

print("\nDataset Shape:", df.shape)
print("\nDataset Info:")
df.info()

# STEP 2: EDA

plt.figure(figsize=(5, 4))
sns.countplot(x="PurchaseStatus", data=df)
plt.title("Purchase vs No Purchase")
plt.show()

plt.figure(figsize=(10, 6))
sns.heatmap(df.select_dtypes(include=np.number).corr(), annot=True, cmap="coolwarm")
plt.title("Feature Correlation Heatmap")
plt.show()

# STEP 3: DATA PREPROCESSING

X = df.drop("PurchaseStatus", axis=1)
y = df["PurchaseStatus"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# STEP 4: MODEL BUILDING

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000),
    "Decision Tree": DecisionTreeClassifier(random_state=42),
    "KNN": KNeighborsClassifier()
}

results = {}

for name, model in models.items():
    model.fit(X_train_scaled, y_train)
    preds = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, preds)
    results[name] = acc

    print(f"\n{name}")
    print("Accuracy:", acc)
    print("Classification Report:\n", classification_report(y_test, preds))
    print("Confusion Matrix:\n", confusion_matrix(y_test, preds))

# STEP 5: RANDOM FOREST WITH HYPERPARAMETER TUNING

rf = RandomForestClassifier(random_state=42)

param_grid = {
    "n_estimators": [100, 200, 300],
    "max_depth": [None, 5, 10, 15],
    "min_samples_split": [2, 5, 10],
    "min_samples_leaf": [1, 2, 4],
    "max_features": ["sqrt", "log2"]
}

grid = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    n_jobs=-1,
    scoring="accuracy"
)

grid.fit(X_train_scaled, y_train)

best_rf = grid.best_estimator_
rf_preds = best_rf.predict(X_test_scaled)
rf_acc = accuracy_score(y_test, rf_preds)

results["Random Forest (Tuned)"] = rf_acc

print("\nRandom Forest (Tuned) Accuracy:", rf_acc)
print("Classification Report:\n", classification_report(y_test, rf_preds))
print("Confusion Matrix:\n", confusion_matrix(y_test, rf_preds))
print("Best Random Forest Parameters:", grid.best_params_)

# STEP 6: MODEL COMPARISON

print("\nModel Comparison (Accuracy):")
for model_name, acc in results.items():
    print(f"{model_name}: {acc:.4f}")

best_model_name = max(results, key=results.get)
print("\nBest Performing Model:", best_model_name)

if "Random Forest" in best_model_name:
    best_model = best_rf
else:
    best_model = models[best_model_name]

# STEP 7: SAVE MODEL AND SCALER

os.makedirs("models", exist_ok=True)

joblib.dump(best_model, "models/best_model.pkl")
joblib.dump(scaler, "models/scaler.pkl")

print(f"\nSaved {best_model_name} model to models/best_model.pkl")
print("Saved scaler to models/scaler.pkl")
