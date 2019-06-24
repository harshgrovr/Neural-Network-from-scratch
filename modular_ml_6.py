import sys

import numpy as np
import pandas as pd
import random

import matplotlib.pyplot as plt
from bokeh.plotting import figure
from bokeh.io import output_notebook, push_notebook, show


def build_Model(feature_set,hidden_nodes,output_label=2):

    hidden_nodes = 5
    output_label = 2
    attributes = feature_set.shape[1]
    model={}
    model['wh1'] = np.random.rand(attributes, hidden_nodes)
    model['bh1'] = np.random.rand(hidden_nodes)
    model['wh2'] = np.random.rand(hidden_nodes, hidden_nodes)
    model['bh2'] = np.random.randn(hidden_nodes)
    model['wo'] =np.random.rand(hidden_nodes,output_label)
    model['bo'] = np.random.rand(output_label)
    return model


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

def feed_forward(model, feature_set):
    wh1,bh1,wh2,bh2,bo,wo = model['wh1'], model['bh1'], model['wh2'], model['bh2'], model['bo'], model['wo']
    #phase 1 feedforward
    zh1 = np.dot(feature_set, wh1) + bh1
    ah1 = sigmoid(zh1)

    # Phase 2 feedforward
    zh2 = np.dot(ah1, wh2) + bh2
    ah2 = sigmoid(zh2)

    # Phase 3 feedforward
    zo = np.dot(ah2, wo) + bo
    ao = softmax(zo)
    return zh1,ah1,zh2,ah2,zo,ao

def backprogation(feature_set, zh1, ah1, zh2, ah2, zo, ao, one_hot_vector, wo, wh2):
    ########## Back Propagation
    ########## Phase 1
    dcost_dzo = ao - one_hot_vector
    dzo_dwo = ah2
    dcost_dwo = np.dot(np.transpose(dzo_dwo), dcost_dzo)
    dcost_bo = dcost_dzo

    ########## Back Propagation
    ########## Phase 2
    dz_ah2 = wo
    dcost_ah2 = np.dot(dcost_dzo, np.transpose(dz_ah2))
    dah2_dzh2 = sigmoid_der(zh2)
    dz_dwh2 = ah1
    dcost_wh2 = np.dot(np.transpose(dz_dwh2), dcost_ah2 * dah2_dzh2)

    ########## Back Propagation
    ########## Phase 3
    dzh1_dwh1 = feature_set
    dah1_dzh1 = sigmoid_der(zh1)
    dcost_ah1 = np.dot(dcost_ah2 * dah2_dzh2, np.transpose(wh2))
    dcost_dbh2 = dcost_ah2 * dah2_dzh2
    dcost_dwh1 = np.dot(np.transpose(dzh1_dwh1), dah1_dzh1 * dcost_ah1)
    dcost_dbh1 = dah1_dzh1 * dcost_ah1
    return dcost_dwh1, dcost_wh2, dcost_dwo, dcost_dbh1, dcost_dbh2, dcost_bo

def calculate_loss(batchX, batchy, ao, model):
    zh1,ah1,zh2,ah2,zo,ao= feed_forward(model, batchX)
    num_example = batchX.shape[0]
    loss = np.sum(-batchy * np.log(ao))
    return 1. / num_example * loss



def train(model, feature_set, one_hot_vector, training_data_x, training_data_y, validation_data_x, validation_data_y, epochs, lr, batchsize):
    losses = []
    flag = 0
    validatinLossHistory = []
    validationAccuracyHistory = []
    trainingAccuracyHistory = []
    wo = model['wo']
    wh2 = model['wh2']
    for epoch in range(epochs):
        if(flag == 1):
            break;
        epochLoss = []
        #feed forward
        # trying SGD
        random_int = np.random.randint(0, feature_set.shape[0]-batchsize)
        XX = np.array([feature_set[random_int: random_int+batchsize]])
        YY = np.array([one_hot_vector[random_int: random_int+batchsize]])
        for batchX, batchY in zip(XX, YY):
            zh1,ah1,zh2,ah2,zo,ao= feed_forward(model, batchX)
            #backprogation
            dcost_dwh1, dcost_wh2, dcost_dwo, dcost_dbh1, dcost_dbh2, dcost_bo = backprogation(batchX,zh1,ah1,zh2,ah2,zo,zo,batchY, wo, wh2)
            #update weights and biases
            model['wh1'] -= lr * dcost_dwh1
            model['wh2'] -= lr * dcost_wh2
            model['wo'] -= lr * dcost_dwo
            model['bh1'] -= lr * np.sum(dcost_dbh1, axis=0)
            model['bh2'] -= lr * np.sum(dcost_dbh2, axis=0)
            model['bo'] -= lr * np.sum(dcost_bo, axis=0)
        if(epoch % 250 == 0):
            print(epoch)
            epoch+=1
            loss=calculate_loss(validation_data_x,validation_data_y,ao,model)
            validatinLossHistory.append(loss)
            trainingAccuracyHistory.append([accuracy(0, model, [], batchX, batchY), epoch])
            validationAccuracyHistory.append([accuracy(0, model, [], validation_data_x, validation_data_y), epoch])
            if (validatinLossHistory.__len__() > 15):
                if(all(i <=loss for i in validatinLossHistory[-15:])):
                    flag=1
                    break
    return model,trainingAccuracyHistory, validationAccuracyHistory

def accuracy(match, model, result, feature_set, one_hot_vector):
    wh1,bh1,wh2,bh2,bo,wo = model['wh1'], model['bh1'], model['wh2'], model['bh2'], model['bo'], model['wo']
    num_of_data = feature_set.shape[0]
    for i in range(num_of_data):
        # Phase 1 feedforward
        zh1 = np.dot(feature_set, wh1) + bh1
        ah1 = sigmoid(zh1)

        # Phase 2 feedforward
        zh2 = np.dot(ah1, wh2) + bh2
        ah2 = sigmoid(zh2)

        # Phase 3 feedforward
        zo = np.dot(ah2, wo) + bo
        ao = softmax(zo)
        if(ao[i][0] > ao[i][1]):
            result.append([1,0])
        else:
            result.append([0, 1])

    for i in range(num_of_data):
        if(np.array_equal(result[i],one_hot_vector[i])):
            match += 1
    return match/(num_of_data)


def divide_dataset(feature_set, one_hot_vector):
    train = int(feature_set.shape[0] * 0.95)
    val = feature_set.shape[0] - train
    training_data_x = np.array(feature_set[0: train])
    training_data_y = np.array(one_hot_vector[0: train])
    validation_data_x = np.array(feature_set[train: train+val])
    validation_data_y = np.array(one_hot_vector[train: train+val])
    return training_data_x, training_data_y, validation_data_x, validation_data_y



def main():
    error_cost = []
    t = 100000
    hidden_nodes = 5
    output_node =2
    np.random.seed(42)
    types_dict = {'height': float, 'weight': float, 'gender': int}
    df = pd.read_csv('DWH_Training.csv', dtype=types_dict)
    df.columns = ['height', 'weight', 'gender']
    df = df.replace(-1, 0)
    # normalize data b/w 0 and 1
    df['height'] = (df['height'] - np.min(df['height'])) / (np.max(df['height']) - np.min(df['height']))
    df['weight'] = (df['weight'] - np.min(df['weight'])) / (np.max(df['weight']) - np.min(df['weight']))
    if (df.columns.size > 3):
        df = df.drop([df.columns[0]], axis=1)
    feature_set = np.array([[df['height'][j + 1], df['weight'][j + 1]] for j in range(df.__len__() - 1)])
    y = np.array([[df['gender'][j + 1] for j in range(df.__len__() - 1)]])
    one_hot_vector = []
    for i in range(df.__len__() - 1):
        if (y[0][i] == 1):
            one_hot_vector.append([1, 0])
        else:
            one_hot_vector.append([0, 1])
    one_hot_vector = np.array(one_hot_vector)
    model = build_Model(feature_set,hidden_nodes,output_node)


    # divide the dataset
    training_data_x, training_data_y, validation_data_x, validation_data_y = divide_dataset(feature_set,one_hot_vector)
    model, trainingAccuracyHistory, validationAccuracyHistory = train(model, training_data_x, training_data_y,training_data_x, training_data_y, validation_data_x, validation_data_y, 100000, lr=5/t, batchsize= 50)


    p = figure(plot_width=400, plot_height=400)
    p.xaxis.axis_label = 'Accuracy'
    p.yaxis.axis_label = 'Epochs'
    print(np.array(trainingAccuracyHistory), np.array(validationAccuracyHistory))
    x=[]
    y=[]
    for i in trainingAccuracyHistory:
        x = i[0]
        y = i[1]

        p.circle(x, y, size=5, color="navy", alpha=0.5, legend="training_accuracy")

    x = []
    y = []
    for i in validationAccuracyHistory:
        x = i[0]
        y = i[1]
        p.circle(x, y, size=5, color="green", alpha=0.5, legend="validation accuracy")
    show(p)

if __name__ == "__main__":
    main()




