import numpy as np

#m = no of training examples
#nh = size of hidden layer
#lets assume
nh = 10
m = 100
learning_rate = 0.01
#after taking input i will flatten it to (4096,m)

def softmax(a):
    return np.exp(a)/np.sum(np.exp(a))


def ini(nh):
    w1 = np.random.rand(nh,4096)*0.01
    b1 = np.zeros((nh,1))
    w2 = np.random.randn(10,nh)*0.01
    b2 = np.zeros((10,1))
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2}
    return parameters


def forward_prop(X, w, b):
    z = np.dot(w, X) + b
    a = softmax(z)
    return a,z


#took the eqn from coursera andrew ng course 1
def back_prop(w, a, da):
    _ = a(1-a)
    dz = np.multiply(da, _)
    dw = np.dot(dz, a.T)/m
    db = np.sum(dz, axis=1, keepdims=True)/m
    da = np.dot(w.T, dz)

    return dw,db,da


def train(iterations, X, Y):
    parameters = ini(nh)
    w1 = parameters["w1"]
    b1 = parameters["b1"]
    w2 = parameters["w2"]
    b2 = parameters["b2"]
    a = np.array(nh, 64)
    for i in range(iterations):
        a, z1 = forward_prop(X, w1, b1)
        yhat, z2 = forward_prop(a, w2, b2)
        da2 = -Y/yhat + (1-Y)/(1-yhat)
        dw2, db2, da2 = back_prop(w2, a, da2)
        dw1, db1, da1 = back_prop(w1, X, da2)
        w1 = w1 - dw1*learning_rate
        b1 = b1 - db1*learning_rate
        w2 = w2 - dw2*learning_rate
        b2 = b2 - db2*learning_rate
    return parameters