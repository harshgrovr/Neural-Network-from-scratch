import numpy as np
import pandas as pd


error_cost=[]
np.random.seed(42)
types_dict = {'height': float, 'weight': float, 'gender': int}
df = pd.read_csv('DWH_Training.csv', dtype=types_dict)
df.columns = ['height','weight','gender']
df = df.replace(-1, 0)


# normalize data b/w 0 and 1

df['height'] = (df['height'] - np.min(df['height']))/(np.max(df['height']) - np.min(df['height']))
df['weight'] = (df['weight'] - np.min(df['weight']))/(np.max(df['weight']) - np.min(df['weight']))


if(df.columns.size > 3):
    df = df.drop([df.columns[0]], axis=1)


feature_set = np.array([[df['height'][j+1], df['weight'][j+1] ] for j in range(df.__len__()-1)])
#print(feature_set)

y = np.array([[df['gender'][j+1] for j in range(df.__len__()-1)]])

#print(y)


hidden_nodes = 5
output_label = 2
attributes = feature_set.shape[1]
#print(attributes)
wh1 = np.random.rand(attributes, hidden_nodes)
#print(wh1[0])
#print(wh1[0])
bh1 = np.random.rand(hidden_nodes)
bh2 = np.random.randn(hidden_nodes)

wh2 = np.random.rand(hidden_nodes, hidden_nodes)
bh2 = np.random.rand(hidden_nodes)

wo = np.random.rand(hidden_nodes,output_label)
bo = np.random.rand(output_label)

lr = 0.0001;
#print(lr)

#print(wh1, wh2, who)

one_hot_vector = []

# print(y[0])
for i in range(df.__len__() - 1):
    if (y[0][i] == 1):
        one_hot_vector.append([1, 0])
    else:
        one_hot_vector.append([0, 1])
one_hot_vector = np.array(one_hot_vector)


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def sigmoid_der(x):
    return sigmoid(x) * (1 - sigmoid(x))


def softmax(A):
    expA = np.exp(A)
    return expA / expA.sum(axis=1, keepdims=True)

for epoch in range(500):
    #print(one_hot_vector)
    # Phase 1 feedforward
    #print(feature_set)
    zh1 = np.dot(feature_set, wh1) + bh1
    #print(feature_set.shape)
    #print(wh1.shape)
    #print(bh1.shape)
    #print(zh1[0])
    ah1 = sigmoid(zh1)
    #print(ah1)
    #print(ah1.shape)
    #print(zh1.shape)
    #print(sigmoid(zh1))



    # Phase 2 feedforward
    zh2 = np.dot(ah1,wh2) + bh2
    #print(zh2)
    #print('\n\n')
    #print(ah1.shape)
    #print(wh2.shape)
    #print(bh2.shape)
    #print(zh2.shape)
    ah2 = sigmoid(zh2)
    #print(zh2)
    #print(ah2.shape)

    # Phase 3 feedforward
    zo = np.dot(ah2,wo) + bo
    #print('\n\n')
    #print(ah2.shape)
    #print(wo.shape)
    #print(bo.shape)
    #print(zo.shape)
    #print(zo)
    ao = softmax(zo)

    #exp_scores = np.exp(zo)
    #out = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    #print(ao)


    ########## Back Propagation
    ########## Phase 1
    dcost_dzo = ao - one_hot_vector
    dzo_dwo = ah2
    dcost_dwo = np.dot(np.transpose(dzo_dwo), dcost_dzo)
    #print(dcost_dzo.shape, dzo_dwo.shape)
    #print(dcost_dwo.shape)
    dcost_bo = dcost_dzo

    ########## Back Propagation
    ########## Phase 2
    dz_ah2 = wo
    dcost_ah2 =  np.dot(dcost_dzo, np.transpose(dz_ah2))
    dah2_dzh2 = sigmoid_der(zh2)
    dz_dwh2 = ah1
    dcost_wh2 = np.dot(np.transpose(dz_dwh2), dcost_ah2 * dah2_dzh2)
    #print(dcost_wh2.shape)

    ########## Back Propagation
    ########## Phase 3
    dzh1_dwh1 = feature_set
    dah1_dzh1 = sigmoid_der(zh1)
    #print(np.transpose(wh1).shape, (dcost_ah2  * dah2_dzh2).shape)
    dcost_ah1 = np.dot(dcost_ah2 * dah2_dzh2, np.transpose(wh2))
    dcost_dbh2 =  dcost_ah2 * dah2_dzh2

    #print(dcost_ah1[0].shape)


    #print(np.transpose(dah1_dzh1) * dcost_ah1)
    dcost_dwh1 = np.dot(np.transpose(dzh1_dwh1), dah1_dzh1 * dcost_ah1)
    #print(dcost_dwh1[0])

    dcost_dbh1 =   dah1_dzh1 * dcost_ah1
    # stochastic gradient descent #Update weights

    dcost_dwh1+= dcost_dwh1
    dcost_wh2+= dcost_wh2
    dcost_dwo+= dcost_dwo

    dcost_dbh1+=dcost_dbh1
    dcost_dbh2+=dcost_dbh2
    dcost_bo+=dcost_bo

    if(epoch % 500 == 0):
        wh1 -= lr * dcost_dwh1
        wh2 -= lr * dcost_wh2
        wo -= lr * dcost_dwo

        bh1 -= lr * np.sum(dcost_dbh1, axis=0)
        bh2 -= lr * np.sum(dcost_dbh2, axis=0)
        bo -= lr * np.sum(dcost_bo, axis=0)
        #print(dcost_dwo)

        dcost_dwh1=0
        dcost_wh2=0
        dcost_dwo=0
        dcost_dbh1=0
        dcost_dbh2=0
        dcost_bo=0


    if(epoch % 500 == 0):
        loss = np.sum(-one_hot_vector * np.log(ao))
        print('Loss function value: ', loss)
        error_cost.append(loss)

result = []

def accuracy(match =0):
    for i in range(df.__len__()-1):
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

    for i in range(df.__len__()-1):
        if(np.array_equal(result[i],one_hot_vector[i])):
            match += 1
    print(match)
    print('accuracy is', match/(df.__len__()-1))

accuracy(0)