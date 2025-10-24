# Import necessary libraries
import numpy as np
import pandas as pd
import regressions as regs
import classifications as cls
from sentence_transformers import SentenceTransformer, util
import plotly.express as px

# Instantiates a pre-trained SentenceTransformer model that will be used for semantic sentence similarity
model = SentenceTransformer ("stsb-mpnet-base-v2")


# takes in a list of values and returns their mean.
def mean(val):
    return sum (val) / len (val)


# takes in a Pandas DataFrame and a list of column names, and returns the mean of the first column.
def auto_mean(data, arr):
    # Extract the values of the first column from the DataFrame
    val = data[arr[0]].values
    # Calculate the mean of the values using the mean() function defined earlier in the code
    return "mean:", mean (val)


# takes in a Pandas DataFrame and a list of column names, and returns the sum of the first column.
def auto_sum(data, arr):
    # Extract the values of the first column from the DataFrame
    val = data[arr[0]].values
    # Calculate the sum of the values using the sum() function
    return "sum:", sum (val)


# takes in a list of values and returns their variance.
def variance(values):
    # Calculate the mean of the values using the mean() function defined earlier in the code
    ans = mean (values)
    n = len (values)
    var = 0

    # Loop over each element in the input list and calculate the variance formula
    for i in values:
        var += (i - ans) ** 2

    # Divide the sum of squared differences by the number of samples minus one to obtain the variance
    var = var / (n - 1)
    return var


# takes in a list of values and returns their standard deviation.
def std(values):
    # Calculate the variance of the values using the variance() function defined earlier in the code, and take the square root
    return variance (values) ** 0.5


# takes in a Pandas DataFrame and a list of column names, and returns the standard deviation of the first column.
def auto_std(data, arr):
    # Extract the values of the first column from the DataFrame
    val = data[arr[0]].values
    # Calculate the standard deviation of the values using the std() function defined earlier in the code
    return "standard deviation:", std (val)


# takes in two lists of numbers and returns the Pearson correlation coefficient for them.
def pearson(X, Y):
    # Copy the input lists into new variables x and y
    x = X
    y = Y

    # Calculate the means of the x and y variables
    x_bar = mean (x)
    y_bar = mean (y)

    numerator = 0
    denom_x = 0
    denom_y = 0

    # Loop over each pair of corresponding elements in x and y, and update the numerator and denominator variables
    for i, j in zip (x, y):
        numerator += (i - x_bar) * (j - y_bar)
        denom_x += (i - x_bar) ** 2
        denom_y += (j - y_bar) ** 2

    # Calculate the denominator of the Pearson correlation coefficient using the square roots of the sums of squared differences
    denominator = (denom_x * denom_y) ** 0.5

    # Return the Pearson correlation coefficient, rounded to three decimal places
    return round (numerator / denominator, 3)


# takes in a number and a list, and returns a list of positions where the number appears in the list.
def pos(num, arr):
    # Initialize an empty list to store the indices where the number appears
    indices = []

    # Loop over each element in the list, and if it matches the input number, append its index to the indices list
    for i in range (len (arr)):
        if arr[i] == num:
            indices.append (i + 1)

    # If there are any indices in the list, return the list
    if indices:
        return indices


# takes in a list of numbers and returns a list of the rank of each number in the list, assuming
# that the list is sorted in descending order.
def rank(data):
    # Initialize an empty list to store the ranks
    ranking = []

    # Create a copy of the input list sorted in descending order
    copy = sorted (data)[::-1]

    # Loop over each number in the input list, and calculate its rank based on its position in the sorted list
    for n in data:
        ranking.append (mean (pos (n, copy)))

    # Return the list of ranks
    return ranking


# Node class which contains a data value and points towards another value
class Node:
    def __init__(self, item):
        self.item = item
        self.next = None


# takes a list of numbers and links each number to its rank
def val_rank_link(array):
    # obtain the ranks for the array values
    x = rank (array)
    link_array = []
    # link each number to its corresponding rank
    for i, j in zip (array, x):
        i = Node (i)
        i.next = j
        link_array.append (i)
    # return the list of linked numbers with their ranks
    return link_array


# takes in a Pandas DataFrame and a list of column names and calculates the Spearman rank correlation coefficient for the first two columns
def auto_spearman(data, arr):
    # get the column names for the two columns
    col1, col2 = arr[0], arr[1]
    # get the number of values in the columns
    n = len (data[col1].values)
    # link the values in the two columns to their ranks
    x = val_rank_link (data[col1].values)
    y = val_rank_link (data[col2].values)
    # calculate the differences between the ranks of each pair of values
    d = [i.next - j.next for i, j in zip (x, y)]
    # square the differences and sum them up
    d_squared = list (map (lambda x: x ** 2, d))
    d_sum = sum (d_squared)
    # calculate the correlation coefficient using the Spearman formula
    correlation = round (1 - ((6 * d_sum) / (n ** 3 - n)), 3)
    # return the correlation coefficient
    return "correlation:", correlation


# takes in a list of numbers and returns the median of the numbers
def median(numbers):
    # Sort the numbers
    sorted_numbers = sorted (numbers)
    # Get the length of the sorted list
    n = len (sorted_numbers)
    # Check if the list has an even number of elements
    if n % 2 == 0:
        # If it does, return the average of the middle two elements
        return (sorted_numbers[n // 2 - 1] + sorted_numbers[n // 2]) / 2
    else:
        # If it doesn't, return the middle element
        return sorted_numbers[n // 2]


# takes in a Pandas DataFrame and a list of column names, and returns the median of the first column.
def auto_median(data, arr):
    # Get the values for the specified column
    val = data[arr[0]].values
    # Compute the median of the values
    return "median:", median (val)


# takes in a list of values and returns the lower quartile
def lower_quartile(numbers):
    # Sort the list of numbers in ascending order
    sorted_numbers = sorted (numbers)
    # Get the length of the list
    n = len (sorted_numbers)
    # Calculate the midpoint index to exclude for even and odd length lists
    if n % 2 == 0:
        exclude = n // 2
    else:
        exclude = n // 2
    # Return the median of the lower half of the list
    return median (sorted_numbers[:exclude])


# takes in a list of values and returns the upper quartile
def upper_quartile(numbers):
    # Sort the list of numbers in ascending order
    sorted_numbers = sorted (numbers)
    # Get the length of the list
    n = len (sorted_numbers)
    # Calculate the midpoint index to exclude for even and odd length lists
    if n % 2 == 0:
        exclude = n // 2
    else:
        exclude = n // 2 + 1
    # Return the median of the upper half of the list
    return median (sorted_numbers[exclude:])


# Takes in a list of numbers and returns a set of the values that are outliers
def outlier(values):
    # Calculate the upper and lower quartiles of the data
    upper = upper_quartile (values)
    lower = lower_quartile (values)
    # Calculate the interquartile range (IQR) of the data
    IQR = upper - lower
    # Calculate the lower and upper bounds for outliers
    lowest = lower - 1.5 * IQR
    highest = upper + 1.5 * IQR
    # Find the values that are outside the lower and upper bounds
    outliers = []
    for i in values:
        if i > highest or i < lowest:
            outliers.append (i)
    # If there are any outliers, return them as a set
    if len (outliers) > 0:
        return set (outliers)


# takes in a Pandas DataFrame and a list of column names, and returns the outliers of the first column.
def auto_outlier(data, arr):
    # Get the column values from the dataset
    val = data[arr[0]].values
    # Calculate the outliers in the column values and return them
    return "outliers:", outlier (val)


# returns a box plot of a given feature in the data
def box_plot(data, arr):
    feature = arr[0]
    return px.box (data_frame=data, x=feature, title=f'Box Plot For {feature}')


# returns a pie chart of a given feature in the data
def pie_chart(data, arr):
    feature = arr[0]
    return px.pie (data, feature, hole=0.3, title=f'Pie Chart For {feature}')


# returns a horizontal bar chart of two given features in the data
def bar_chart(data, arr):
    x, y = arr[0], arr[1]
    return px.bar (data, x, y, orientation='h', title=f'Bar Chart For {x} and {y}')


# returns a scatter plot of two given features in the data
def scatter_graph(data, arr):
    x, y = arr[0], arr[1]
    return px.scatter (data, x, y, title=f'Scatter Graph for {x} and {y}')


# returns a line plot of two given features in the data
def line_graph(data, arr):
    x, y = arr[0], arr[1]
    return px.line (data, x, y, title=f'Line Graph for {x} and {y}')


# returns a histogram of a given feature in the data
def histogram(data, arr):
    feature = arr[0]
    return px.histogram (data, feature, title=f'Histogram For {feature}')


# creating a list of available commands for program
corpus = ["calculate mean",  # command to calculate mean
          "calculate sum",  # command to calculate sum
          "calculate standard deviation",  # command to calculate standard deviation
          "calculate correlation between _ and _",  # command to calculate correlation
          "calculate median for _",  # command to calculate median
          "calculate outliers for _",  # command to calculate outliers
          "make box plot for _",  # command to create a box plot
          "make bar chart for _",  # command to create a bar chart
          "make pie chart for _",  # command to create a pie chart
          "make scatter graph for _",  # command to create a scatter plot
          "make line graph for _",  # command to create a line plot
          "make histogram for _"]  # command to create a histogram

# creating a dictionary of some numbers as the keys and some functions as the values
commands = {0: auto_mean,  # command to calculate mean
            1: auto_sum,  # command to calculate sum
            2: auto_std,  # command to calculate standard deviation
            3: auto_spearman,  # command to calculate correlation
            4: auto_median,  # command to calculate median
            5: auto_outlier,  # command to calculate outliers
            6: box_plot,  # command to create a box plot
            7: bar_chart,  # command to create a bar chart
            8: pie_chart,  # command to create a pie chart
            9: scatter_graph,  # command to create a scatter plot
            10: line_graph,  # command to create a line plot
            11: histogram}  # command to create a histogram


# takes in a text input and the columns of a dataset and returns a number based on which command the text input was
# most similar to and returns two names of columns from the dataset depending on if they appear in the text input
def fake_nlp(input, attributes):
    # encode the list of commands into embeddings and convert them to tensors
    corpus_embeddings = model.encode (corpus, convert_to_tensor=True)

    # encode the list of columns into embeddings and convert them to tensors
    attribute_embeddings = model.encode (attributes, convert_to_tensor=True)

    # encode the input text into embeddings and convert it to a tensor
    input_embeddings = model.encode (input, convert_to_tensor=True)

    # calculate the cosine similarity between the input and the list of commands
    similarity_1 = util.pytorch_cos_sim (input_embeddings, corpus_embeddings)[0]

    # calculate the cosine similarity between the input and the list of columns
    similarity_2 = util.pytorch_cos_sim (input_embeddings, attribute_embeddings)[0]

    # get the indices of the two columns with the highest similarity to the input
    two_attributes = np.argpartition (-similarity_2, range (2))[0:2]

    # return the index of the command with the highest similarity and the two most similar columns
    return int (np.argmax (similarity_1)), two_attributes.tolist ()


# takes in a list of the predicted values and actual values and returns the rmse of this
def rmse(predictions, targets):
    return np.sqrt (((predictions - targets) ** 2).mean ())


# takes in two lists of binary values, returns how many values in the predicted set is equal to value in the true
# set as a ratio of the total number of values in the true set.
def accuracy(pred, true):
    correct = sum (1 for p, t in zip (pred, true) if p == t)
    accurate = correct / len (pred)
    return accurate * 100


# calculates linear regression based on epoch, learning rate, and data inputs
# returns a string representing the model and its coefficients, and the root mean squared error (RMSE) of the model's prediction
def calc_lin_reg(epoch, lr, data):
    # extract independent variable x and dependent variable y from input data
    x = data[data.columns.values[0]]
    y = data[data.columns.values[1]]
    # create a LinearRegression object with specified hyperparameters and fit it to the data
    modem = regs.LinearRegression (lr, epoch)
    coefficients = modem.fit (x, y)
    # generate a string representation of the linear model with its coefficients
    string = modem.series (coefficients)
    # make predictions based on the fitted model and calculate its RMSE
    prediction = modem.predict (x)
    accurate = rmse (prediction, y)
    return string, accurate


# calculates polynomial regression based on degree of polynomial and data inputs
# returns a string representing the model and its coefficients, and the RMSE of the model's prediction
def calc_poly_reg(degree, data):
    # extract independent variable x and dependent variable y from input data
    x = data[data.columns.values[0]]
    y = data[data.columns.values[1]]
    # create a PolynomialRegression object with specified degree of polynomial and fit it to the data
    switch = regs.PolynomialRegression (degree)
    coefficients = switch.fit (x, y)
    # generate a string representation of the polynomial model with its coefficients
    string = switch.series (coefficients)
    # generate a function based on the polynomial model and make predictions based on it, then calculate its RMSE
    function = switch.evaluation (string)
    prediction = switch.predict (function, x)
    accurate = rmse (prediction, y)
    return string, accurate


# calculates logistic regression based on epoch, learning rate, data inputs, and target column name
# returns a list of predicted values, the accuracy of the model's prediction, and a mapping of binary values to target variable values
def calc_log_reg(epoch, lr, data, target):
    # create a LogisticRegression object with specified hyper-parameters and preprocess data and target column
    server = cls.LogisticRegression (lr, epoch)
    x, y = server.pre_process (data, target)
    # fit the model to the preprocessed data and predict values based on it
    server.fit (x, y)
    preds = server.predict (x)
    # calculate the accuracy of the model's prediction and return the predicted values, accuracy, and target map
    accurate = accuracy (preds, y)
    return preds, accurate, server.target_map


# calculates decision tree classifier based on minimum split, maximum depth, data inputs, and target column name
# returns a list of predicted values, the accuracy of the model's prediction, and a mapping of binary values to target variable values
def calc_dec_tree(min_split, max_depth, data, target):
    # create a DecisionTree object with specified hyper-parameters and preprocess data and target column
    backbone = cls.DecisionTree (min_split, max_depth)
    x, y = backbone.pre_process (data, target)
    y = y.astype ('int64')
    # fit the model to the preprocessed data and predict values based on it
    backbone.fit (x, y)
    preds = backbone.predict (x)
    # calculate the accuracy of the model's prediction and return the predicted values, accuracy, and target map
    accurate = accuracy (preds, y)
    return preds, accurate, backbone.target_map
