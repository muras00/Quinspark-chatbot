import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


print("Loading Dataset for Clustering...")
file_path = "Disease_symptom_and_patient_profile_dataset.csv"
data = pd.read_csv(file_path)


print("Preprocessing Data...")
data = data.dropna()
features = ['Fever', 'Cough', 'Fatigue', 'Difficulty Breathing', 'Age', 'Gender', 'Blood Pressure', 'Cholesterol Level']
binary_map = {'Yes': 1, 'No': 0}
data.replace(binary_map, inplace=True)


data['Gender'] = data['Gender'].map({'Male': 1, 'Female': 0})
data['Blood Pressure'] = data['Blood Pressure'].map({'Low': 0, 'Normal': 1, 'High': 2})
data['Cholesterol Level'] = data['Cholesterol Level'].map({'Low': 0, 'Normal': 1, 'High': 2})


scaler = StandardScaler()
X_scaled = scaler.fit_transform(data[features])


print("Applying K-Means Clustering...")
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(X_scaled)


print("Visualizing Clusters...")
plt.figure(figsize=(10, 6))
sns.scatterplot(x=data['Age'], y=data['Cholesterol Level'], hue=data['Cluster'], palette="deep", s=100)
plt.title("K-Means Clustering: Age vs Cholesterol Level")
plt.xlabel("Age")
plt.ylabel("Cholesterol Level")
plt.legend(title="Cluster")
plt.show()

print("Clustering Analysis Complete!")
