#!/usr/bin/env python
# CS 534
# AI1 skeleton code
# By Sean Chu
import csv
import numpy
import numpy as np
import matplotlib.pyplot as plt

means = list()
stds = list()


# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    with open(path, 'r') as input:
        read_data = csv.reader(input)
        loaded_data = [row for row in read_data]
        return loaded_data

# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(data, normalize, drop_sqrt_living15):
    # Your code here:
    preprocessed_data = data.copy()
    #preprocessed_data[0][14] = 'age_since_renovated'
    preprocessed_data.pop(0)

    if drop_sqrt_living15:
        preprocessed_data = modify_features(preprocessed_data)

    for i in range(len(preprocessed_data)):
        year = float(preprocessed_data[i][1].split('/')[2])
        month = float(preprocessed_data[i][1].split('/')[0])
        date = float(preprocessed_data[i][1].split('/')[1])
        preprocessed_data[i][14] = year - float(preprocessed_data[i][13]) if float(preprocessed_data[i][14]) == 0 else year - float(preprocessed_data[i][14])
        preprocessed_data[i] = [1] + [year] + [month] + [date] + preprocessed_data[i][2:]
        preprocessed_data[i] = [float(n) for n in preprocessed_data[i]]
        #print(preprocessed_data[i])

    preprocessed_data = numpy.matrix(preprocessed_data)

    if normalize:
        preprocessed_data = preprocessed_data.transpose()
        a = numpy.matrix([[15, 31], [2, 6], [1, 3]])
        #a[0] = [d+1 for d in a[0]]
        #print(a[0])

        if not means:
            for i in range(preprocessed_data.shape[0]):
                if not drop_sqrt_living15 and (i != 0 and i != 9 and i != 22):
                    means.append(numpy.mean(preprocessed_data[i]))
                    stds.append(numpy.std(preprocessed_data[i]))
                    #print(means[-1], stds[-1])

                elif drop_sqrt_living15 and (i != 0 and i != 9 and i != 21):
                    means.append(numpy.mean(preprocessed_data[i]))
                    stds.append(numpy.std(preprocessed_data[i]))

                else:
                    means.append('blank')
                    stds.append('blank')

        for i in range(preprocessed_data.shape[0]):
            if not drop_sqrt_living15 and (i != 0 and i != 9 and i != 22):
                for j in range(preprocessed_data[i].shape[1]):
                    preprocessed_data[i, j] = (preprocessed_data[i, j] - means[i]) / stds[i]

            elif drop_sqrt_living15 and (i != 0 and i != 9 and i != 21):
                for j in range(preprocessed_data[i].shape[1]):
                    preprocessed_data[i, j] = (preprocessed_data[i, j] - means[i]) / stds[i]

        preprocessed_data = preprocessed_data.transpose()

    print(preprocessed_data[0])

    return preprocessed_data

# Implements the feature engineering required for part 4. Quite similar to preprocess_data.
# Expand the arguments of this function however you like to control which feature modification
# approaches are / aren't active.
def modify_features(data):
    # Your code here:
    modified_data = list()

    for i in range(len(data)):
        modified_data.append(data[i][0:19] + data[i][20:])

    return modified_data

# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gd_train(data, labels, lr):
    # Your code here:
    losses = list()
    loss = 0.
    criteria = 10**-6
    weights = np.matrix([1. for i in range(data.shape[1])])
    m = np.matrix(np.zeros(data.shape[1]))
    #print((weights * data[0].T - labels[0]) * data[0])

    for count in range(4000):
        for i in range(data.shape[0]):
            m += (weights * data[i].T - labels[i]) * data[i]
            loss += (weights * data[i].T - labels[i])[0, 0]**2

        m *= (2. / data.shape[0])
        weights -= lr * m
        loss /= data.shape[0]
        losses.append(loss)

        if len(losses) > 1 and (loss - losses[-2] >= 0. or abs(loss - losses[-2]) <= criteria):
            print(count + 1, weights, loss, sep='\n')
            return weights, losses

        m = np.matrix(np.zeros(data.shape[1]))
        loss = 0.

    print(4000, weights, losses[-1], sep='\n')

    return weights, losses

# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses):
    # Your code here:
    plt.figure(figsize = (8,6))
    plt.plot([i for i in range(1, len(losses) + 1)], losses)
    plt.scatter([i for i in range(1, len(losses) + 1)], losses, marker='o', color='red')
    plt.title("Iteration v.s. MSE")
    plt.xlabel("Iteration")
    plt.ylabel("MSE")
    plt.show()

    return

# Invoke the above functions to implement the required functionality for each part of the assignment.

# Part 0  : Data preprocessing.
# Your code here:
training_data = load_data('IA1_train.csv')
valid_data = load_data('IA1_dev.csv')

processed_tr_da = preprocess_data(training_data, True, False)
processed_vl_da = preprocess_data(valid_data, True, False)
a = np.matrix([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
print(a[:, -1])
# Part 1 . Implement batch gradient descent and experiment with different learning rates.
# Your code here:''''''
weights, losses = gd_train(processed_tr_da[:, :processed_tr_da.shape[1] - 1], processed_tr_da[:, -1], 0.1)
#print(losses)
plot_losses(losses)

mse = 0.
w_pro_vl_da = processed_vl_da[:, :processed_vl_da.shape[1] - 1]
y_pro_vl_da = processed_vl_da[:, -1]

for i in range(w_pro_vl_da.shape[0]):
    mse += (weights * w_pro_vl_da[i].T - y_pro_vl_da[i])[0, 0] ** 2

mse /= w_pro_vl_da.shape[0]
print(mse)
means.clear()
stds.clear()


# Part 2 a. Training and experimenting with non-normalized data.
# Your code here:''''''
processed_tr_da = preprocess_data(training_data, False, False)
processed_vl_da = preprocess_data(valid_data, False, False)
weights, losses = gd_train(processed_tr_da[:, :processed_tr_da.shape[1] - 1], processed_tr_da[:, -1], 10**-11)
plot_losses(losses)
mse = 0.
w_pro_vl_da = processed_vl_da[:, :processed_vl_da.shape[1] - 1]
y_pro_vl_da = processed_vl_da[:, -1]

for i in range(w_pro_vl_da.shape[0]):
    mse += (weights * w_pro_vl_da[i].T - y_pro_vl_da[i])[0, 0] ** 2

mse /= w_pro_vl_da.shape[0]
print(mse)


# Part 2 b Training with redundant feature removed. 
# Your code here:''''''
processed_tr_da = preprocess_data(training_data, True, True)
processed_vl_da = preprocess_data(valid_data, True, True)
weights, losses = gd_train(processed_tr_da[:, :processed_tr_da.shape[1] - 1], processed_tr_da[:, -1], 0.0001)
#print(losses)
plot_losses(losses)

mse = 0.
w_pro_vl_da = processed_vl_da[:, :processed_vl_da.shape[1] - 1]
y_pro_vl_da = processed_vl_da[:, -1]

for i in range(w_pro_vl_da.shape[0]):
    mse += (weights * w_pro_vl_da[i].T - y_pro_vl_da[i])[0, 0] ** 2

mse /= w_pro_vl_da.shape[0]
print(mse)
means.clear()
stds.clear()
