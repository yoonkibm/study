import math

def logistic_regression(w, b, x, y, lr):
    z = sum(w_i*x_i for w_i, x_i in zip(w, x)) + b
    y_hat = 1 / (1 + math.exp(-z))

    loss = 0.5*(y_hat - y)**2

    dL_dz = (y_hat-y) * y_hat * (1 - y_hat)

    grad_w = [dL_dz * x_i for x_i in x]
    grad_b = dL_dz

    print(f"ŷ   = {y_hat:.3f}")
    print(f"L   = {loss:.3f}")
    print(f"dL/dw = {[round(g, 3) for g in grad_w]}")
    print(f"dL/db = {grad_b:.3f}")

    w = w - [w_ * lr for w_ in grad_w]
    b = b - [b_ * lr for b_ in grad_b]

    return w, b

w = [0.3, -0.2]
b = 0.1
x = [2, -1]
y = 1


for i in range(10):
    w, b = logistic_regression(w, b, x, y, 0.001)

z = sum(w_i * x_i for w_i, x_i in zip(w, x)) + b
y_hat = 1 / (1 + math.exp(-z))

print(y_hat, y)