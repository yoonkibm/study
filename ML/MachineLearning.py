import numpy as np

class LinearRegression:
    def __init__(self):
        super().__init__()
        self.w = None
    
    def fit(self, X, y, method='gd', max_iter=50, lr=0.001):
        """
        method option: 'gd' gradient descent
                        'normal_eq' normal equation
        max_iter: 최대 반복 횟수
        lr: 학숩률 alpha
        """
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        self.w = np.zeros(X.shape[1])
        m = len(y)
        if method == 'gd':
            for i in range(max_iter):
                y_pred = X.dot(self.w)

                grad = (2/m)*(X.T.dot(y_pred-y))

                self.w -=  lr*grad
                if i % 10 == 0:
                    mse = np.mean((y_pred - y)**2)
                    print("epoch: {}/{}\tmse:{}".format(i, max_iter,mse))

        else:
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X.dot(self.w)
    
    def test(self, X, y, measure='mse'):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        y_pred = X.dot(self.w)
        if measure == 'mse':
            mse = np.mean((y_pred-y)**2)
            print("mse:{}".format(mse))

class LogisticRegression:
    def __init__(self):
        super().__init__()
        self.w = None

    def fit(self, X, y, max_iter=50, lr=0.001):
        X = np.hstack([np.ones((X.shape[0], 1)), X])

        self.w = np.zeros(X.shape[1])
        m = len(y)
        for i in range(max_iter):
            z = X.dot(self.w)
            y_pred = 1/(1+np.exp(-z))

            grad = (1/m)*(X.T.dot(y_pred - y))

            self.w -= lr*grad

            if i % 10 == 0:
                ce = np.mean(-1*(y*np.log(y_pred + 1e-9)+(1-y)*np.log(1-y_pred + 1e-9)))
                print("epoch: {}/{}\tbinary cross entropy:{}".format(i, max_iter, ce))

    def predict(self, X):
        if X.ndim == 1:
            X = X[np.newaxis, :]  # 혹은 X.reshape(1, -1)
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        prob = 1/(1+np.exp(-1*X.dot(self.w)))
        return (prob >= 0.5).astype(int)

    def test(self, X, y):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        y_pred = 1/(1+np.exp(-1*X.dot(self.w)))
        ce = np.mean(-1*(y*np.log(y_pred)+(1-y)*np.log(1-y_pred)))
        print("Binary cross entropy:{}".format(ce))