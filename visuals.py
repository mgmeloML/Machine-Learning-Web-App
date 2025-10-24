# Import necessary libraries
import plotly.graph_objs as go
import plotly.express as px
import numpy as np

# Takes in a string that represents a mathematical function (something like f(x) = 3x + 2) and a value for x,
# converts the string to a Python expression, and evaluates it with the given x value
def evaluation(string, x):
    # Remove unnecessary text from the string and replace ^ with **
    string = string[9:-1]
    string = string.replace("^", "**")
    string = string.replace("{", "")
    string = string.replace("}", "")
    # Replace 'x' with '*x' to properly multiply variables
    string = string.replace("x", "*x")
    # Evaluate the expression with the given x value
    return eval(string)

# Takes in a dataframe and a string representing a linear regression model,
# generates a scatter plot of the data, and overlays a line representing the linear regression model
def regline(data, string):
    # Get the column names for x and y from the dataframe
    x = data.columns.values[0]
    y = data.columns.values[1]
    # Generate the scatter plot
    fig = px.scatter(data, x=x, y=y)
    # Generate a set of x values to plot the regression line with
    start = min(data[x])
    stop = max(data[x])
    x_form = np.linspace(start, stop)
    # Generate the regression line and add it to the plot
    fig.add_trace(go.Scatter(
        x=x_form,
        y=evaluation(string, x_form),
        name='Regression Line',
        line=dict(color='red')))
    return fig

# Takes in binary predicted and actual values, generates a confusion matrix
# returns a plotly figure representing the matrix
def confusion_matrix(predicted, actual):
    # Initialize the matrix
    conf_matrix = [[0, 0], [0, 0]]

    # Fill in the matrix by comparing predicted and actual values
    for i in range(len(predicted)):
        if predicted[i] == 1 and actual[i] == 1:
            conf_matrix[0][0] += 1
        elif predicted[i] == 0 and actual[i] == 0:
            conf_matrix[1][1] += 1
        elif predicted[i] == 1 and actual[i] == 0:
            conf_matrix[0][1] += 1
        elif predicted[i] == 0 and actual[i] == 1:
            conf_matrix[1][0] += 1

    # Generate the confusion matrix plot
    fig = px.imshow(conf_matrix, text_auto=True, aspect='auto',
                     labels=dict(x='True Class', y='Predicted Class'),
                     x=['True Positive', 'True Negative'],
                     y=['Predicted Positive', 'Predicted Negative'])
    fig.update_xaxes(side="top")

    return fig
