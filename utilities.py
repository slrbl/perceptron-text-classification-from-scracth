# Author: walid.daboubi@gmail.com
# Version: 1.0 - 2017/12/24
# About: perceptron algorithm applied on sentiment analysis

import numpy as np
from random import randint
import math

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict(x):
    if x <= 0.5:
        return 0
    else:
        return 1

def clean_text(data, remove_list):
    cleaned_text = []
    for data_unit in data:
        for char in remove_list:
            data_unit = data_unit.replace(char, '')
        data_unit = data_unit.lower()
        cleaned_text.append(data_unit)
    return cleaned_text

def shuffle_data(data_list, shuffle_factor):
    m = len(data_list)
    for i in range(shuffle_factor):
        rndIndex = randint(0, m - 1)
        tmp = data_list[rndIndex]
        rndIndex2 = randint(0, len(data_list) - 1)
        data_list[rndIndex] = data_list[rndIndex2]
        data_list[rndIndex2] = tmp
    return data_list

# Get data and features (word bag)
def get_data(data_list, start, end, word_bag):
    phrases=[]
    get_w_b = False
    if word_bag == None:
        word_bag = []
        get_w_b = True
    i=0;
    m = end - start
    Y = np.zeros(shape=(m, 1))
    for data_unit in data_list[start:end - 1]:
        splitted_data_unit = data_unit.split('\t')
        # Get Y
        Y[i, 0] = int(splitted_data_unit[0])
        words = splitted_data_unit[1].split(' ')
        if get_w_b == True:
            for word in words:
                if word not in word_bag:
                    word_bag.append(word)
        i+=1
    # Get X
    X = np.zeros(shape=(len(Y), len(word_bag)))
    i = 0;
    for data_unit in data_list[start:end - 1]:
        phrases.append(data_unit)
        splitted_data_unit = data_unit.split('\t')[1].split(' ')
        x = [0] * len(word_bag)
        word_count = {}
        for word in splitted_data_unit:
            if get_w_b == False:
                if word in word_bag:
                    if word not in word_count:
                        word_count[word] = 1.
                    else:
                        word_count[word] += 1.
            else:
                if word not in word_count:
                    word_count[word] = 1.
                else:
                    word_count[word] += 1.
        for word in word_count:
            if word in word_bag:
                x[word_bag.index(word)] = word_count[word]
        X[i]=(x)
        i += 1
    if get_w_b == True:
        return word_bag, X, Y,phrases
    else:
        return X, Y,phrases

def train_model(m, ITERATIONS, LEARNING_RATE, Y, X):
    X = X.T
    Y = Y.T
    # weights vector initialization
    w = np.zeros((X.shape[0], 1))

    # Weights random initialization
    for j in range(X.shape[0]):
        w[j, 0] = randint(-10, 10) * 0.01
    # Bias initialization
    b = 0
    # Cost initialization
    cost = 0
    for i in range(ITERATIONS):
        A = sigmoid(np.dot(w.T, X) + b)
        cost = (-1. / m) * np.sum((Y * np.log(A)) + (1. - Y) * np.log(1. - A))
        print "Iteration: "+str(i)+", cost: "+str(cost)
        if math.isnan(cost):
            return w, b
        dw = (1. / m) * np.dot(X, (A - Y).T)
        db = (1. / m) * np.sum(A - Y)
        w = w - dw * LEARNING_RATE
        b = b - db * LEARNING_RATE
    return w,b

def get_precision_stats(predicted, actual):
    TP, FP, TN, FN = 0, 0, 0, 0
    # Result precision tatistics
    for i in range(len(actual)):
        if predict(predicted[i]) == 1:
            # True positive: predicted is 1 and real is 1
            if actual[i] == 1:
                TP += 1.
            # False positive: predicted is 1 and real is 0
            elif actual[i] == 0:
                FP += 1.
        elif predict(predicted[i]) == 0:
            # True negative: predicted is 0 and real is 0
            if actual[i] == 0:
                TN += 1.
            # False negative: predicted is 0 and real is 1
            elif actual[i] == 1:
                FN += 1.
    accuracy = (TP + TN) / (TP + TN + FP + FN)
    recall = TP / ( TP + FN )
    precision = TP / ( TP + FP )
    f1 = (2 * recall * precision) / (recall + precision)
    return TP, FP, TN, FN, accuracy, recall, precision, f1
