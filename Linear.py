import numpy as np


class LinearRegression:

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
            y_predict = np.dot(X, self.weight) + self.bias  # y=x.w+b
            dw = (1/n_sample)*np.dot(X.T, (y_predict-y))  # (1/N)*2x(y_predict-y_true)
            db = (1/n_sample)*np.sum(y_predict-y)
            # update weight,bias

            self.weight-= self.lr * dw
            self.bias -= self.lr * db

    def predict(self, X):
        y_predict = np.dot(X, self.weight) + self.bias
        return y_predict
from sklearn.model_selection import train_test_split
from sklearn import datasets


X, y = datasets.make_regression(n_samples=100, n_features=1, random_state=4, noise=20)
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=1234)

model = LinearRegression()
model.fit(X_train, Y_train)
p = model.predict(X_test)


def mse(y_test, predict):
    return np.mean((y_test-predict) ** 2)

mse_value = mse(Y_test,p)
print(mse_value)