# CS 534
# AI1 skeleton code
# By Quintin Pope
import pandas as pd
import numpy as np

# Loads a data file from a provided file location.
def load_data(path):
    # Your code here:
    loaded_data = pd.read_csv(path)
    #print(data_train)
    return loaded_data
    


# Implements dataset preprocessing, with boolean options to either normalize the data or not, 
# and to either drop the sqrt_living15 column or not.
#
# Note that you will call this function multiple times to generate dataset versions that are
# / aren't normalized, or versions that have / lack sqrt_living15.
def preprocess_data(data, normalize, drop_sqrt_living15):
    # Your code here:
    #drop id
    data.drop(columns=['id'], inplace=True)

    #split the date and drop data value
    data = pd.merge((data['date'].str.split('/', expand = True)), data,
    how='left', left_index= True, right_index= True)
    
    data.drop(columns=['date'], inplace=True)
    data.rename(columns={0:'month', 1:'day', 2:'year'}, inplace=True)
    data[['month', 'day', 'year']] = data[['month', 'day', 'year']].astype('int')
    data.insert(loc=0, column='dummy', value=1)
    
    #change renovate year
    for i in range(data.shape[0]):
        if data['yr_renovated'][i] == 0:
            data['yr_renovated'][i] = data['year'][i] - data['yr_built'][i]
        else:
            data['yr_renovated'][i] = data['year'][i] - data['yr_renovated'][i]
    
    data.rename(columns={'yr_renovated':'age_science_renovated'}, inplace=True)
    
    #normalize
    if normalize == False:
        features = [f for f in data.columns if f not in ['dummy', 'waterfront', 'price']]
        for feature in features:
            current_mean = data[feature].mean()
            current_std = data[feature].std()
            for i in range(data.shape[0]):
                data[feature][i] = (data[feature][i] - current_mean)/current_std
    
    normalize = True


    preprocessed_data = data
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
d = load_data('IA1_train.csv')
print(d.info())
f = preprocess_data(d,False,1)
print(f.head(10))

# Part 1 . Implement batch gradient descent and experiment with different learning rates.
# Your code here:


# Part 2 a. Training and experimenting with non-normalized data.
# Your code here:


# Part 2 b Training with redundant feature removed. 
# Your code here:



