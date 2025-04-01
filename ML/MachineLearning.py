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
        
        if method == 'gd':
            m = len(y)
            for i in range(max_iter):
                y_pred = X.dot(self.w)

                grad = (2/m)*X.T.dot(y_pred-y)

                self.w -=  lr*grad
                if i % 10 == 0:
                    y_pred_prelearn = X.dot(self.w)
                    mse = np.mean((y_pred_prelearn-y)**2)
                    print("epoch: {}/{}\tmse:{}".format(i, max_iter,mse))

        else:
            self.w = np.linalg.inv(X.T @ X) @ X.T @ y

    def predict(self, X):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        return X.dot(self.w)
    
    def test(self, X, y, measure='mse'):
        X = np.hstack([np.ones((X.shape[0], 1)), X])
        m = len(y)
        y_pred = X.dot(self.w)
        if measure == 'mse':
            mse = np.mean((y_pred-y)**2)
            print("mse:{}".format(mse))