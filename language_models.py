import numpy as np

#m = no of training examples
#nh,nh2 = size of hidden layer
#lets assume
nh = 10
m = 100
learning_rate = 0.01
#after taking input i will flatten it to (4096,m)

def softmax(a):
    return np.exp(a)/np.sum(np.exp(a))

def relu(a):
    Z = np.zeros(a.shape)
    return np.max(a,Z)


def ini(nh,nh2):
    w1 = np.random.rand(nh,4096)*0.01
    b1 = np.zeros((nh,1))
    w2 = np.random.randn(nh2,nh)*0.01
    b2 = np.zeros((nh2,1))
    w3 = np.random.randn(10,nh2)*0.01
    b3 = np.random.randn((10,1))
    parameters = {"w1": w1,
                  "b1": b1,
                  "w2": w2,
                  "b2": b2,
                  "w3": w3,
                  "b3": b3}
    return parameters


def forward_prop(X, w, b, i):
    z = np.dot(w, X) + b
    if i == 1:
        a = softmax(z)
    else:
        a = relu(z)
    return a,z


#took the eqn from coursera andrew ng course 1
def back_prop(w, a, da,i):
    if i == 1:
        _ = a(1-a)
    else:
        _ = np.where(a > 0 , 1, 0)
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
    w3 = parameters["w3"]
    b3 = parametera["b3"]
    a = np.array(nh, 64)
    for i in range(iterations):
        a1, z1 = forward_prop(X, w1, b1,0)
        a2, z2 = forward_prop(a1, w2, b2,0)
        yhat, z3 = forward_prop(a2, w3, b3, 1)
        da4 = -Y/yhat + (1-Y)/(1-yhat)
        dw3, db3, da3 = back_prop(w3, a2, da4,1)
        dw2, db2, da2 = back_prop(w2, a1, da3,0)
        dw1, db1, da1 = back_prop(w1, X, da2,0)
        w1 = w1 - dw1*learning_rate
        b1 = b1 - db1*learning_rate
        w2 = w2 - dw2*learning_rate
        b2 = b2 - db2*learning_rate
        w3 = w3 - dw3*learning_rate
        b3 = b3 - db3*learning_rate
    return parameters
