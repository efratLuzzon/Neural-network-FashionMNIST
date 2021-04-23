import sys
import scipy
from scipy.special import expit, logit
import numpy as np

NUM_EPOCH = 56
LEARNING_RATE = 0.2
FIRST_LAYER = 90
SEOND_LAYER = 10




def sigmoid(x):
    # x = x - np.max(x)
    return expit(x)


def normalize(list_to_normal):
    return list_to_normal / 255


def initialize_neural_network_variable():
    W1 = np.random.rand(FIRST_LAYER, 784)  # * 0.05
    b1 = np.random.rand(FIRST_LAYER, 1)  # * 0.05
    W2 = np.random.rand(SEOND_LAYER, FIRST_LAYER)  # * 0.05
    b2 = np.random.rand(SEOND_LAYER, 1)  # * 0.05
    dict_variable = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}
    return dict_variable


def my_shuffle(train_x, train_y):
    indices = np.arange(train_x.shape[0])
    np.random.shuffle(indices)
    X_train = train_x[indices]
    Y_train = train_y[indices]
    return X_train, Y_train


def fprop(x, y, dict_variable, test_mode):
    x = x.reshape(784, 1)
    W1, b1, W2, b2 = [dict_variable[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x) + b1
    z1 /= 784
    h1 = sigmoid(z1)

    z2 = np.dot(W2, h1) + b2
    z2 /= FIRST_LAYER
    h2 = scipy.special.softmax(z2)

    if test_mode:
        return h2
    ret = {'x': x, 'y': y, 'h1': h1, 'h2': h2}
    for key in dict_variable:
        ret[key] = dict_variable[key]
    return ret


def bprop(fprop_cache):
    x, y, h1, h2 = [fprop_cache[key] for key in ('x', 'y', 'h1', 'h2')]

    y_array = np.zeros((10, 1))
    y_array[int(y)] = 1
    dz2 = h2 - y_array
    dw2 = np.dot(dz2, h1.T)
    db2 = dz2

    dz1 = np.dot(fprop_cache['W2'].T, dz2) * h1 * (1 - h1)
    dw1 = np.dot(dz1, x.T)
    db1 = dz1

    return {'b1': db1, 'W1': dw1, 'b2': db2, 'W2': dw2}


def update_dict_variable(dict_variable, bprop_cache):
    dict_variable['W1'] = dict_variable['W1'] - LEARNING_RATE * bprop_cache['W1']
    dict_variable['b1'] = dict_variable['b1'] - LEARNING_RATE * bprop_cache['b1']

    dict_variable['W2'] = dict_variable['W2'] - LEARNING_RATE * bprop_cache['W2']
    dict_variable['b2'] = dict_variable['b2'] - LEARNING_RATE * bprop_cache['b2']
    return dict_variable


def algo_neural_network(dict_variable, train_x, train_y):
    for i in range(NUM_EPOCH):
        X_train, Y_train = my_shuffle(train_x, train_y)
        for j in range(len(X_train)):
            x = X_train[j]
            y = Y_train[j]
            fprop_cache = fprop(x, y, dict_variable, False)
            bprop_cache = bprop(fprop_cache)
            dict_variable = update_dict_variable(dict_variable, bprop_cache)
    return dict_variable




def test(dict_variable, train_x, train_y):
    print("start\n")
    count = 0
    for x, y in zip(train_x, train_y):
        h2 = fprop(x, y, dict_variable, True)
        y_hat = np.argmax(h2)
        if y == y_hat:
            count += 1
        print("y " + str(y) + "y_hat " + str(y_hat))
    #RESULT.append(count / len(train_y))


def print_result(dict_variable, test_x):
    file = open("test_y", "w")
    for x in test_x:
        h2 = fprop(x, None, dict_variable, True)
        y_hat = np.argmax(h2)
        file.write(f"{int(y_hat)}\n")
    file.close()


def main():
    args = sys.argv
    train_x = args[1]
    train_y = args[2]
    test_x = args[3]

    train_x = np.loadtxt(train_x)
    train_y = np.loadtxt(train_y)
    train_x = normalize(train_x)
    test_x = np.loadtxt(test_x)
    test_x = normalize(test_x)
    dict_variable = initialize_neural_network_variable()
    dict_variable = algo_neural_network(dict_variable, train_x, train_y)
    print_result(dict_variable, test_x)


if __name__ == "__main__":
    main()
