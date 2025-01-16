from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.tree import export_text, plot_tree
import matplotlib.pyplot as plt
import pandas as pd


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

# Visualisasi pohon keputusan dalam bentuk teks
tree_rules = export_text(decision_tree, feature_names=list(X.columns))
print(tree_rules)

# Visualisasi pohon keputusan dalam bentuk diagram
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, feature_names=X.columns, class_names=['No Death', 'Death'], filled=True)
plt.show()