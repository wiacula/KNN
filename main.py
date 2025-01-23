import numpy as np
from collections import Counter
from sklearn import datasets, metrics
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import unittest

# Implementacja własnego KNN
class KNN:
    def __init__(self, k=3):
        self.k = k

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y

    def predict(self, X):
        return np.array([self._predict(x) for x in X])

    def _predict(self, x):
        # Oblicz odległości euklidesowe do wszystkich punktów treningowych
        distances = [np.linalg.norm(x - x_train) for x_train in self.X_train]
        # Wybierz k najbliższych sąsiadów
        k_indices = np.argsort(distances)[:self.k]
        k_nearest_labels = [self.y_train[i] for i in k_indices]
        # Zwróć najczęściej występującą etykietę
        most_common = Counter(k_nearest_labels).most_common(1)
        return most_common[0][0]

# Wczytaj dane MNIST (cyfry odręczne)
digits = datasets.load_digits()
data = digits.images.reshape((len(digits.images), -1))  # Spłaszcz obrazy 8x8 do wektora

# Podziel dane na zbiory treningowe i testowe
X_train, X_test, y_train, y_test = train_test_split(data, digits.target, test_size=0.5, shuffle=False)

# Tworzenie i trenowanie własnego KNN
knn_custom = KNN(k=3)
knn_custom.fit(X_train, y_train)
predictions_custom = knn_custom.predict(X_test)

# Tworzenie i trenowanie KNN z biblioteki scikit-learn
knn_sklearn = KNeighborsClassifier(n_neighbors=3)
knn_sklearn.fit(X_train, y_train)
predictions_sklearn = knn_sklearn.predict(X_test)

# Wyświetlenie raportów klasyfikacji
print("Custom KNN Classification Report:\n")
print(metrics.classification_report(y_test, predictions_custom))

print("Scikit-learn KNN Classification Report:\n")
print(metrics.classification_report(y_test, predictions_sklearn))

# Wizualizacja macierzy pomyłek
metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictions_custom)
plt.title("Confusion Matrix - Custom KNN")
plt.show()

metrics.ConfusionMatrixDisplay.from_predictions(y_test, predictions_sklearn)
plt.title("Confusion Matrix - Scikit-learn KNN")
plt.show()

# Wizualizacja przykładowych predykcji
_, axes = plt.subplots(1, 5, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test[:5], predictions_custom[:5]):
    ax.set_axis_off()
    ax.imshow(image.reshape(8, 8), cmap="gray", interpolation="nearest")
    ax.set_title(f"Pred: {prediction}")
plt.suptitle("Sample Predictions - Custom KNN")
plt.show()

_, axes = plt.subplots(1, 5, figsize=(10, 3))
for ax, image, prediction in zip(axes, X_test[:5], predictions_sklearn[:5]):
    ax.set_axis_off()
    ax.imshow(image.reshape(8, 8), cmap="gray", interpolation="nearest")
    ax.set_title(f"Pred: {prediction}")
plt.suptitle("Sample Predictions - Scikit-learn KNN")
plt.show()

# Testy jednostkowe
class TestKNN(unittest.TestCase):
    def setUp(self):
        # Dane testowe do testów jednostkowych
        self.knn = KNN(k=3)
        self.X_train = np.array([[1, 2], [2, 3], [3, 3], [6, 5], [7, 7], [8, 6]])
        self.y_train = np.array([0, 0, 0, 1, 1, 1])
        self.knn.fit(self.X_train, self.y_train)

    def test_predict_single(self):
        # Test predykcji dla pojedynczego punktu
        point = np.array([5, 5])
        prediction = self.knn._predict(point)
        self.assertEqual(prediction, 1)

    def test_predict_multiple(self):
        # Test predykcji dla wielu punktów
        points = np.array([[5, 5], [2, 2]])
        predictions = self.knn.predict(points)
        self.assertTrue(np.array_equal(predictions, [1, 0]))

if __name__ == "__main__":
    unittest.main()
