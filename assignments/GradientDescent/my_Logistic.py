import numpy as np
import pandas as pd

class my_Logistic:

    def __init__(self, learning_rate=0.1, batch_size=10, max_iter=100, shuffle=False):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_iter = max_iter
        self.shuffle = shuffle
        self.w = None
        self.w0 = None

    def sigmoid(self, z):
        return 1 / (1 + np.exp(-z))

    def fit(self, X, y):
        data = X.to_numpy()
        targets = np.array(y)
        n_samples, n_features = data.shape

        
        self.w = np.zeros(n_features)
        self.w0 = 0

        # Gradient Descent
        for _ in range(self.max_iter):
            if self.shuffle:
                indices = np.arange(n_samples)
                np.random.shuffle(indices)
                data = data[indices]
                targets = targets[indices]

            for start in range(0, n_samples, self.batch_size):
                end = start + self.batch_size
                batch_x = data[start:end]
                batch_y = targets[start:end]

                
                model = np.dot(batch_x, self.w) + self.w0
                predictions = self.sigmoid(model)

                
                error = predictions - batch_y
                dw = np.dot(batch_x.T, error) / len(batch_x)
                dw0 = np.sum(error) / len(batch_x)

               
                self.w -= self.learning_rate * dw
                self.w0 -= self.learning_rate * dw0

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.to_numpy()
        z = np.dot(X, self.w) + self.w0
        return self.sigmoid(z)

    def predict(self, X):
        probs = self.predict_proba(X)
        return np.round(probs).astype(int)

