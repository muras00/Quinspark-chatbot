import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib


print("Loading and Cleaning Dataset...")
file_path = "Disease_symptom_and_patient_profile_dataset.csv"
data = pd.read_csv(file_path)


data = data.dropna()
print(f"Dataset Loaded! Shape: {data.shape}")
print(f"Columns: {list(data.columns)}")


binary_cols = ["Fever", "Cough", "Fatigue", "Difficulty Breathing"]
for col in binary_cols:
    data[col] = data[col].map({'Yes': 1, 'No': 0})


label_encoder = LabelEncoder()
data['Blood Pressure'] = label_encoder.fit_transform(data['Blood Pressure'])
data['Cholesterol Level'] = label_encoder.fit_transform(data['Cholesterol Level'])


data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Outcome Variable'] = data['Outcome Variable'].map({'Positive': 1, 'Negative': 0})


X = data.drop(columns=['Disease', 'Outcome Variable'])  
y = data['Outcome Variable']


scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


joblib.dump(scaler, "scaler.pkl")
print("Scaler saved as 'scaler.pkl'")


print("\nBalancing Classes with SMOTE...")
smote = SMOTE(random_state=42, k_neighbors=1)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y)
print("Class Distribution After SMOTE:")
print(pd.Series(y_resampled).value_counts())


print("\nSplitting Data into Train and Test Sets...")
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)


print("\nInitializing Models...")
models = {
    "Random Forest": {
        "model": RandomForestClassifier(random_state=42),
        "params": {
            'n_estimators': [100, 200],
            'max_depth': [5, 10, None],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
    },
    "Gradient Boosting": {
        "model": GradientBoostingClassifier(random_state=42),
        "params": {
            'n_estimators': [100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5]
        }
    },
    "SVM": {
        "model": SVC(probability=True, random_state=42),
        "params": {
            'C': [1, 10],
            'kernel': ['linear']
        }
    }
}


best_model = None
best_accuracy = 0
results = {}
best_models = {}  

for name, model_details in models.items():
    print(f"\nTraining {name} with Hyperparameter Tuning...")
    model = model_details['model']
    param_grid = model_details['params']
    grid_search = GridSearchCV(model, param_grid, cv=3, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)

    
    best_estimator = grid_search.best_estimator_
    y_pred = best_estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, best_estimator.predict_proba(X_test)[:, 1])

    print(f"Best Parameters for {name}: {grid_search.best_params_}")
    print(f"{name} Accuracy: {accuracy * 100:.2f}%")
    print(f"{name} ROC-AUC Score: {roc_auc:.2f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=1))

    
    results[name] = {"Accuracy": accuracy, "ROC-AUC": roc_auc}
    best_models[name] = best_estimator

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model = best_estimator


print("\nSaving the Best Model...")
joblib.dump(best_model, "best_enhanced_symptom_model.pkl")
print(f"Best Model Saved as 'best_enhanced_symptom_model.pkl'")


print("\nVisualizing Confusion Matrices for Each Model...")
for name, model in best_models.items():
    print(f"Confusion Matrix for {name}...")
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", xticklabels=["Negative", "Positive"],
                yticklabels=["Negative", "Positive"])
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()


results_df = pd.DataFrame(results).T
print("\nModel Performance:")
print(results_df)


joblib.dump(results, "model_comparison_results.pkl")
print("Model comparison results saved as 'model_comparison_results.pkl'.")


joblib.dump(best_models, "best_models.pkl")
print("Best models saved as 'best_models.pkl'.")


plt.figure(figsize=(10, 6))
results_df.plot(kind='bar', title="Model Comparison: Accuracy vs ROC-AUC", figsize=(10, 6))
plt.ylabel("Score")
plt.legend()
plt.show()


print("\nFeature Importance for Tree-Based Models:")
tree_models = ["Random Forest", "Gradient Boosting"]
for name in tree_models:
    model = best_models[name]
    importances = model.feature_importances_
    features = X.columns
    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances, y=features)
    plt.title(f"Feature Importance - {name}")
    plt.xlabel("Importance")
    plt.ylabel("Features")
    plt.show()

print("\nModel Comparison and Visualization Complete!")
