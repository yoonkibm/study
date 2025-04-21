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
        n, d = X.shape
        k = Y.shape[1]
        self.num_classes = k

        x_ = np.hstack([np.ones((n, 1)), X])
        d_plus_1 = d + 1

        self.w = np.zeros((d_plus_1, k))
        
        for i in range(max_iter):
            z = x_.dot(self.w)

            exp_z = np.exp(z)

            sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)

            Y_pred = exp_z / sum_exp_z

            loss = -np.mean(np.sum(Y * np.log(Y_pred + 1e-9), axis=1))

            grad = (1.0 / n) * x_.T.dot(Y_pred - Y)

            self.w -= lr*grad

            if i % 10 == 0:
                print("epoch: {}/{}\tCross entropy:{}".format(i, max_iter, loss))

    def predict(self, X):

        if X.ndim == 1:
            X = X.reshape(1, -1)

        n = X.shape[0]
        x_ = np.hstack([np.ones((n, 1)), X])
        z = x_.dot(self.w)
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
        prob = exp_z / sum_exp_z
        return np.argmax(prob, axis=1)

    def test(self, X, Y):
        n = X.shape[0]
        x_ = np.hstack([np.ones((n, 1)), X])
        z = x_.dot(self.w)
        exp_z = np.exp(z)
        sum_exp_z = np.sum(exp_z, axis=1, keepdims=True)
        Y_pred = exp_z / sum_exp_z
        loss = -np.mean(np.sum(Y * np.log(Y_pred + 1e-9), axis=1))
        print("Cross entropy:{}".format(loss))

class NearestNeighborClassification:
    def __init__(self):
        super().__init__()
        self.X = None
        self.Y = None

    def fit(self, args_x, args_y):
        self.X = args_x
        self.Y = args_y

    def predict(self, X_test, k = 3):
        X_train_squared = np.sum(self.X**2, axis=1)
        X_test_squared = np.sum(X_test**2, axis=1)
        cross = X_test @ self.X.T

        dists = np.sqrt(X_test_squared[:, None] + X_train_squared[None, :] - 2*cross)

        knn_indices = np.argpartition(dists, kth = k, axis = 1)[:, :k]

        knn_labels = self.Y[knn_indices]

        preds = []
        for row in knn_labels:
            vals, counts = np.unique(row, return_counts=True)
            preds.append(vals[np.argmax(counts)])
        
        return np.preds(preds)
    
class NearestNeighborRegression:
    def __init__(self):
        super().__init__()
        self.X = None
        self.Y = None

    def fit(self, args_x, args_y):
        self.X = args_x
        self.Y = args_y

    def predict(self, X_test, k = 3):

        if k <= 0 or k > self.X.shape[0]:
            raise ValueError("k must be between 1 and the number of training samples")
        
        X_train_squared = np.sum(self.X**2, axis=1)
        X_test_squared = np.sum(X_test**2, axis=1)
        cross = X_test @ self.X.T

        dists = np.sqrt(X_test_squared[:, None] + X_train_squared[None, :] - 2*cross)

        knn_indices = np.argpartition(dists, kth = k, axis = 1)[:, :k]

        knn_values = self.Y[knn_indices]
        preds = np.mean(knn_values, axis=1)
        
        return preds
    
class GaussianNBClassification:
    def __init__(self):
        super().__init__()
        self.classes = None
        self.class_means = {}
        self.class_vars = {}
        self.class_priors = {}

    def fit(self, X, Y):

        self.classes = np.unique(Y)

        for class_ in self.classes:
            class_data = X[Y == class_]
            self.class_means[class_] = class_data.mean(axis = 0)
            self.class_vars[class_] = np.where(class_data.var(axis = 0) == 0, 1e-9, class_data.var(axis = 0, ddof=0))
            self.class_priors[class_] = len(class_data)/len(X)
    
    def predict(self, X_test):

        proba = []

        for class_ in self.classes:
            mean = self.class_means[class_]
            var = self.class_vars[class_]

            log_prior = np.log(self.class_priors[class_])

            log_posterior = -0.5 * np.log(2 * np.pi * var) - ((X_test - mean)**2)/(2*var)
            log_prob = log_prior + np.sum(log_posterior, axis=1)

            proba.append(log_prob)

        proba = np.array(proba).T
        return np.argmax(proba, axis=1)
        


 
def one_hot_encoding(y, num_classes=None):

    y = np.array(y)

    if num_classes == None:
        num_classes = y.max() + 1

    labels = np.zeros((y.shape[0], num_classes))

    labels[np.arange(y.shape[0]), y] = 1
    
    return labels

def accuracy(Y, Y_pred):
    count_matches = np.count_nonzero(Y == Y_pred)

    accuracy = (count_matches / len(Y))*100
    print("Accuracy: {:.2f}".format(accuracy))
