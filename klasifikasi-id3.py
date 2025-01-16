from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import export_text, plot_tree
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

# Membaca dataset
file_path = 'dataset_update_pasien_gagal_jantung.csv'
dataset = pd.read_csv(file_path)

# Memisahkan fitur dan target
X = dataset.drop(columns=['DEATH_EVENT'])
y = dataset['DEATH_EVENT']

# Membagi dataset menjadi data latih dan data uji (80% latih, 20% uji)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Membuat model pohon keputusan
decision_tree = DecisionTreeClassifier(random_state=42)
decision_tree.fit(X_train, y_train)

# Melakukan prediksi pada data uji
y_pred = decision_tree.predict(X_test)

# Evaluasi model
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Menampilkan hasil evaluasi
print(f"Akurasi: {accuracy * 100:.2f}%")
print("Laporan Klasifikasi:")
print(classification_rep)

# Definisi parameter yang akan diuji
param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid Search
grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train, y_train)

# Hasil terbaik
best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# Evaluasi model terbaik
y_pred_optimized = best_model.predict(X_test)
accuracy_optimized = accuracy_score(y_test, y_pred_optimized)
classification_rep_optimized = classification_report(y_test, y_pred_optimized)

print(f"Parameter Terbaik: {best_params}")
print(f"Akurasi Setelah Optimasi: {accuracy_optimized * 100:.2f}%")
print("Laporan Klasifikasi Setelah Optimasi:")
print(classification_rep_optimized)

# Mengambil pentingnya fitur
feature_importances = best_model.feature_importances_
features = X.columns

# Membuat plot fitur penting
sorted_idx = np.argsort(feature_importances)[::-1]
plt.figure(figsize=(10, 6))
plt.barh(features[sorted_idx], feature_importances[sorted_idx], color='skyblue')
plt.xlabel("Importance")
plt.ylabel("Features")
plt.title("Feature Importances")
plt.gca().invert_yaxis()
plt.show()