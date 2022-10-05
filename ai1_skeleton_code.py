# CS 534
# AI1 skeleton code
# By Quintin Pope
import csv
import copy
#import numpy as np


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
    preprocessed_data[0][14] = 'age_since_renovated'
    for i in range(1, len(data)):
        year = float(data[i][1].split('/')[2])
        month = float(data[i][1].split('/')[0])
        date = float(data[i][1].split('/')[1])
        preprocessed_data[i][14] = year - float(preprocessed_data[i][13]) if float(preprocessed_data[i][14]) == 0 else year - float(preprocessed_data[i][14])
        preprocessed_data[i] = [1] + [year] + [month] + [date] + preprocessed_data[i][2:]
        preprocessed_data[i] = [float(n) for n in preprocessed_data[i]]
        print(preprocessed_data[i])

    if not normalize:
        for i in range(len(data[1])):
            col = list()

            for j in range(1, len(data)):
            col = [data[j][i]

    return preprocessed_data

# Implements the feature engineering required for part 4. Quite similar to preprocess_data.
# Expand the arguments of this function however you like to control which feature modification
# approaches are / aren't active.
def modify_features(data):
    # Your code here:

    return modified_data

# Trains a linear model on the provided data and labels, using the supplied learning rate.
# weights should store the per-feature weights of the learned linear regression.
# losses should store the sequence of MSE losses for each epoch of training, which you will then plot.
def gd_train(data, labels, lr):
    # Your code here:

    return weights, losses

# Generates and saves plots of the training loss curves. Note that you can interpret losses as a matrix
# containing the losses of multiple training runs and then put multiple loss curves in a single plot.
def plot_losses(losses):
    # Your code here:

    return

# Invoke the above functions to implement the required functionality for each part of the assignment.

# Part 0  : Data preprocessing.
# Your code here:
training_data = load_data('IA1_train.csv')
valid_data = load_data('IA1_dev.csv')

preprocess_data(training_data, False, False)

# Part 1 . Implement batch gradient descent and experiment with different learning rates.
# Your code here:


# Part 2 a. Training and experimenting with non-normalized data.
# Your code here:


# Part 2 b Training with redundant feature removed. 
# Your code here:



