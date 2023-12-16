import pandas as pd
import numpy as np
from collections import Counter

class my_KNN:

    def __init__(self, n_neighbors=5, metric="minkowski", p=2):
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.p = p

    def fit(self, X, y):
        self.X_train = X
        self.y_train = y
        self.classes_ = sorted(list(set(y)))

    def predict(self, X):
        predictions = [self._predict(x) for x in X.to_numpy()]
        return predictions

    def predict_proba(self, X):
        proba_list = []
        for x in X.to_numpy():
            neighbors = self._find_neighbors(x)
            counts = Counter(neighbors)
            proba = {cls: counts.get(cls, 0) / self.n_neighbors for cls in self.classes_}
            proba_list.append(proba)
        return pd.DataFrame(proba_list, columns=self.classes_)

    def _find_neighbors(self, x):
        distances = [self.cal_distance(x, x_train) for x_train in self.X_train.to_numpy()]
        k_indices = np.argsort(distances)[:self.n_neighbors]
        return [self.y_train[i] for i in k_indices]

    def _predict(self, x):
        neighbors = self._find_neighbors(x)
        most_common = Counter(neighbors).most_common(1)
        return most_common[0][0]

    def cal_distance(self, x1, x2):
        if self.metric == "euclidean":
            return self._euclidean_distance(x1, x2)
        elif self.metric == "manhattan":
            return self._manhattan_distance(x1, x2)
        elif self.metric == "minkowski":
            return self._minkowski_distance(x1, x2, self.p)
        elif self.metric == "cosine":
            return self._cosine_distance(x1, x2)
        else:
            raise ValueError("Unknown metric")

    def _euclidean_distance(self, x1, x2):
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _manhattan_distance(self, x1, x2):
        return np.sum(abs(x1 - x2))

    def _minkowski_distance(self, x1, x2, p):
        return np.sum(abs(x1 - x2) ** p) ** (1 / p)

    def _cosine_distance(self, x1, x2):
        dot_product = np.dot(x1, x2)
        norm_x1 = np.linalg.norm(x1)
        norm_x2 = np.linalg.norm(x2)
        return 1 - dot_product / (norm_x1 * norm_x2)


