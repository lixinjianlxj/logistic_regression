#!/usr/bin/env python
# encoding: utf-8
# @Time    : 2019/8/15 12:23 AM
# @Author  : Li Xinjian  
# @File    : 01_homework.py.py

import numpy as np
import h5py
import matplotlib.pyplot as plt
from lr_utils import load_dataset


# 训练集是图片，包含猫(y=1)，非猫(y=0)
# 建立logistic回归分类器


# 加载数据集
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()


# 训练集，测试集的个数
m_train = train_set_x_orig.shape[0]
m_test = test_set_x_orig.shape[0]


# 展开成列向量
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T
# print train_set_x_flatten.shape


# 归一化
train_set_x = train_set_x_flatten/255.
test_set_x = test_set_x_flatten/255.
# print train_set_x


# 定义sigmoid函数
def sigmoid(z):

    a = 1 / (1 + np.exp(-z))

    return a


# print sigmoid(np.array([0, 2]))


# 初始化参数
def initialize_with_zeros(dim):

    w = np.zeros((dim, 1))
    b = 0

    return w, b


# 前向传播，反向传播
def propagate(w, b, X, Y):

    m = X.shape[1]  # m是训练样本的个数

    # print m
    A = sigmoid(np.dot(w.T, X) + b)

    cost = np.sum(np.multiply(Y, np.log(A)) + np.multiply((1 - Y), np.log(1 - A))) / -m
    cost = np.squeeze(cost)

    dz = A - Y

    dw = np.dot(X, dz.T) / m

    db = np.sum(dz) / m

    grads = {'dw': dw, 'db': db}

    return grads, cost


# w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1, 2], [3, 4]]), np.array([[1, 0]])
# grads, cost = propagate(w, b, X, Y)
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))
# print ("cost = " + str(cost))
# print w.shape


# 优化
def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost=False):

    costs = []

    for i in range(num_iterations):
        grads, cost = propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)

        if print_cost and i % 100 == 0:
            print 'Cost after iteration %i: %f' % (i, cost)

    params = {'w': w, 'b': b}
    grads = {'dw': dw, 'db': db}

    return params, grads, costs


# params, grads, costs = optimize(w, b, X, Y, num_iterations=100, learning_rate=0.009, print_cost=False)
# print ("w = " + str(params["w"]))
# print ("b = " + str(params["b"]))
# print ("dw = " + str(grads["dw"]))
# print ("db = " + str(grads["db"]))


# 预测
def predict(w, b, X):

    m = X.shape[1]

    Y_prediction = np.zeros((1, m))

    A = sigmoid(np.dot(w.T, X) + b)

    for i in range(m):
        if A[0, i] <= 0.5:
            Y_prediction[0, i] = 0
        else:
            Y_prediction[0, i] = 1

    return Y_prediction


# print ("predictions = " + str(predict(params['w'], params['b'], X)))


# 把模型完整的写到一个函数
def model(X_train, Y_train, X_test, Y_test, num_iterations=2000, learning_rate=0.5, print_cost=False):

    dim = X_train.shape[0]
    w, b = initialize_with_zeros(dim)

    parameters, grads, costs = optimize(w, b, X_train, Y_train, num_iterations, learning_rate, print_cost)

    w = parameters['w']
    b = parameters['b']

    Y_prediction_test = predict(w, b, X_test)
    Y_prediction_train = predict(w, b, X_train)

    print("train accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_train - Y_train)) * 100))
    print("test accuracy: {} %".format(100 - np.mean(np.abs(Y_prediction_test - Y_test)) * 100))

    d = {"costs": costs,
         "Y_prediction_test": Y_prediction_test,
         "Y_prediction_train": Y_prediction_train,
         "w": w,
         "b": b,
         "learning_rate": learning_rate,
         "num_iterations": num_iterations}
    return d


# d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=2000, learning_rate=0.005, print_cost=True)
#
# costs = d['costs']
# plt.figure()
# plt.plot(costs)
# plt.ylabel('cost')
# plt.xlabel('iterations (per hundreds)')
# plt.title("Learning rate =" + str(d["learning_rate"]))
# plt.savefig('cost.png')
# plt.show()


learning_rates = [0.01, 0.005, 0.001, 0.0005, 0.0001]
models = {}
for i in learning_rates:
    print ("learning rate is: " + str(i))
    models[str(i)] = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iterations=1500, learning_rate=i, print_cost=False)
    print ('\n' + "-------------------------------------------------------" + '\n')

for i in learning_rates:
    plt.plot(np.squeeze(models[str(i)]["costs"]), label=str(models[str(i)]["learning_rate"]))

plt.ylabel('cost')
plt.xlabel('iterations')

legend = plt.legend(loc='upper center', shadow=True)
frame = legend.get_frame()
frame.set_facecolor('0.90')
plt.savefig('learning rate')
plt.show()
