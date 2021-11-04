import numpy as np


class Logistic_Regression:

    def __init__(self, lr=0.01, n_iter=1000):
        self.lr = lr
        self.n_iter = n_iter
        self.weight = None
        self.bias = None

    def fit(self, X, y):
        # Initialize parameter
        n_sample, n_feature = X.shape
        self.weight = np.zeros(n_feature)
        self.bias = 0

        for _ in range(self.n_iter):
            linear = np.dot(X, self.weight) + self.bias  # y=x.w+b
            y_predict = self._sigmoid(linear)
            dw = (1 / n_sample) * np.dot(X.T, (y_predict - y))  # (1/N)*2x(y_predict-y_true)
            db = (1 / n_sample) * np.sum(y_predict - y)
            self.weight -= self.lr * dw
            self.bias -= self.lr * db

    def _sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def predict(self, X):
        linear = np.dot(X, self.weight) + self.bias  # y=x.w+b
        y_predict = self._sigmoid(linear)
        y_predict_cl = [1 if i > 0.5 else 0 for i in y_predict]
        return y_predict_cl
from sklearn import datasets
from sklearn.model_selection import train_test_split
data = datasets.load_breast_cancer()
X, y = data.data, data.target

X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1234)


Log = Logistic_Regression(lr=0.001, n_iter=1000)
Log.fit(X_train,Y_train)
p = Log.predict(X_test)


def accuracy(y_true, y_predict):
    acc = np.sum(y_true == y_predict) / len(y_true)
    return acc


a = accuracy(Y_test, p)
print(a)