import numpy as np
import pandas as pd


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

def calculate_loss(feature_set, one_hot_vector, ao):
    num_example = feature_set.shape[0]
    loss = np.sum(-one_hot_vector * np.log(ao))
    return 1. / num_example * loss


def train(model, feature_set, one_hot_vector, epochs, lr):
    losses = []
    previous_loss =float('inf')
    wo = model['wo']
    wh2 = model['wh2']
    for epoch in range(epochs-1):
        #feed forward
        zh1,ah1,zh2,ah2,zo,ao= feed_forward(model, feature_set)
        #backprogation
        dcost_dwh1, dcost_wh2, dcost_dwo, dcost_dbh1, dcost_dbh2, dcost_bo = backprogation(feature_set,zh1,ah1,zh2,ah2,zo,zo,one_hot_vector, wo, wh2)
        #update weights and biases
        model['wh1'] -= lr * dcost_dwh1
        model['wh2'] -= lr * dcost_wh2
        model['wo'] -= lr * dcost_dwo
        model['bh1'] -= lr * np.sum(dcost_dbh1, axis=0)
        model['bh2'] -= lr * np.sum(dcost_dbh2, axis=0)
        model['bo'] -= lr * np.sum(dcost_bo, axis=0)
        if(epoch % 500 == 0):
            loss=calculate_loss(feature_set,one_hot_vector,ao)
            print(epoch)
            print('Loss function value: ', loss)


def accuracy(match, model, result, feature_set, one_hot_vector):
    wh1,bh1,wh2,bh2,bo,wo = model['wh1'], model['bh1'], model['wh2'], model['bh2'], model['bo'], model['wo']
    num_of_data = feature_set.shape[0]
    for i in range(num_of_data-1):
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

    for i in range(num_of_data-1):
        if(np.array_equal(result[i],one_hot_vector[i])):
            match += 1
    print(match)
    print('accuracy is', match/(num_of_data-1))



def main():
    error_cost = []
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
    try:
        model, losses= train(model, feature_set, one_hot_vector, 50000,lr=0.0001)
    except:
        print('An exception occured')

    accuracy(0,model,[],feature_set,one_hot_vector)

if __name__ == "__main__":
    main()
