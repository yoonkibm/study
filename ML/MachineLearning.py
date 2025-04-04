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
        self.num_classes = None

    def fit(self, X, Y, max_iter=50, lr=0.01):
        '''
        X: (N, d)
        Y: (N, k)
        '''
        N, d = X.shape
        K = Y.shape[1]
        self.num_classes = K

        X_ = np.hstack([np.ones((N, 1)), X])
        d_plus_1 = d + 1

        self.w = np.zeros((d_plus_1, K))
        
        for i in range(max_iter):
            Z = X_.dot(self.w)

            expZ = np.exp(Z)

            sumExpZ = np.sum(expZ, axis=1, keepdims=True)

            Y_pred = expZ / sumExpZ

            loss = -np.mean(np.sum(Y * np.log(Y_pred + 1e-9), axis=1))

            grad = (1.0 / N) * X_.T.dot(Y_pred - Y)

            self.w -= lr*grad

            if i % 10 == 0:
                print("epoch: {}/{}\tCross entropy:{}".format(i, max_iter, loss))

    def predict(self, X):

        if X.ndim == 1:
            X = X.reshape(1, -1)

        N = X.shape[0]
        X_ = np.hstack([np.ones((N, 1)), X])
        Z = X_.dot(self.w)
        expZ = np.exp(Z)
        sumExpZ = np.sum(expZ, axis=1, keepdims=True)
        prob = expZ / sumExpZ
        return np.argmax(prob, axis=1)

    def test(self, X, Y):
        N = X.shape[0]
        X_ = np.hstack([np.ones((N, 1)), X])
        Z = X_.dot(self.w)
        expZ = np.exp(Z)
        sumExpZ = np.sum(expZ, axis=1, keepdims=True)
        Y_pred = expZ / sumExpZ
        loss = -np.mean(np.sum(Y * np.log(Y_pred + 1e-9), axis=1))
        print("Cross entropy:{}".format(loss))

def one_hot_encoding(y, num_classes=None):

    y = np.array(y)

    if num_classes == None:
        num_classes = y.max() + 1

    labels = np.zeros((y.shape[0], num_classes))

    labels[np.arange(y.shape[0]), y] = 1
    
    return labels