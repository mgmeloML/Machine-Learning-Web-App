# Import necessary libraries
import numpy as np
from collections import Counter


# Base class for the classification models
class Classification:

    def __init__(self):

        self.target_map = {}

    # Function that preprocesses the input data by converting non-numeric values to numeric values
    # This is necessary so that algorithms can work properly
    def pre_process(self, tabular, target):

        # Separating the target column and the analysis columns
        analysis_set = tabular.drop([target], axis=1)
        target_set = tabular[target]

        # Converting non-numeric values to numeric values
        for i in analysis_set.columns.values:
            # Only non-numeric values need to be converted
            if analysis_set[i].values.dtype != 'int64' and analysis_set[i].values.dtype != 'float64':
                remap = {}

                # Mapping non-numeric values to a numeric value
                for j in tabular[i]:
                    if j not in remap:
                        remap[j] = len(remap)

                # Replacing the non-numeric values with the mapped numeric value
                f = lambda x: remap[x]
                analysis_set[i] = analysis_set[i].apply(f)

        # Converting the target set to numeric values
        for k in target_set.values:
            if k not in self.target_map:
                self.target_map[k] = len(self.target_map)

        # Replacing the original target set with the new numeric target set
        for p, n in enumerate(target_set.values):
            target_set.values[p] = self.target_map[n]

        # Creating a reverse mapping of the numeric target set to the original target set
        self.target_map = {v: k for k, v in self.target_map.items()}

        # Returning the preprocessed data
        return np.array(analysis_set), np.array(target_set)


# Subclass of Classification implementing Logistic regression
class LogisticRegression (Classification):
    # Constructor to initialize the learning rate, epoch, weights, bias, and to inherit from the Classification class
    def __init__(self, learning_rate, epoch):
        self.learning_rate = learning_rate
        self.epoch = epoch
        self.weights = None
        self.bias = None
        super ().__init__ ()

    # Uses the pre_process method in the base class to pre-process the data
    def pre_process(self, data, target):
        return super ().pre_process (data, target)

    # Sigmoid function to be used in the prediction step
    def sigmoid(self, x):
        return 1 / (1 + np.exp (-x))

    # Function to fit the model to the data
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.weights = np.zeros (n_features)
        self.bias = 0
        # loops for the number of epochs
        for i in range (self.epoch):
            # linear regression formulation
            linear_exp = np.dot (X, self.weights) + self.bias
            # linear formula passed into sigmoid function
            predictions = self.sigmoid (linear_exp.astype ('float64'))

            # gradient for weights is calculated
            w_grad = (1 / n_samples) * np.dot (X.T, (predictions - y))
            # gradient for bias calculated
            b_grad = (1 / n_samples) * np.sum (predictions - y)

            # weights are updated with gradient descent
            self.weights = self.weights - self.learning_rate * w_grad
            # bias updated with gradient descent
            self.bias = self.bias - self.learning_rate * b_grad

    # Function to predict the class of a given set of inputs
    def predict(self, X):
        linear_exp = np.dot (X, self.weights) + self.bias
        y_pred = self.sigmoid (linear_exp.astype ('float64'))
        # if value is greater than 0.5 then it is a 1 and if less than 0.5 then it is a 0
        class_pred = [0 if y <= 0.5 else 1 for y in y_pred]
        return class_pred


class DecisionNode:
    # A class to represent a node in the decision tree

    def __init__(self, feature=None, threshold=None, left=None, right=None, *, value=None):
        # Initialize the node with its feature, threshold, left and right child nodes,
        # and a value if it is a leaf node.

        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value

    def is_leaf_node(self):
        # Check if the node is a leaf node by checking if it has a value.

        return self.value is not None

# Subclass of Classification implementing Decision tree
class DecisionTree (Classification):

    def __init__(self, min_samples_split, max_depth, n_features=None):
        # Initialize the DecisionTree object with the minimum number of samples required to split a node,
        # maximum depth of the tree, and number of features to consider at each split (default to None)
        self.min_samples_split = min_samples_split
        self.max_depth = max_depth
        self.n_features = n_features
        self.root = None
        super ().__init__ ()

    # Uses the pre_process method in the base class to pre-process the data
    def pre_process(self, data, target):
        return super ().pre_process (data, target)

    def fit(self, X, y):
        # Determine the number of features to use based on the number of columns in the input matrix X
        # and the specified number of features to consider
        self.n_features = X.shape[1] if not self.n_features else min (X.shape[1], self.n_features)
        # Build the decision tree by recursively splitting nodes based on the input data and labels
        self.root = self.build_tree (X, y)

    def build_tree(self, X, y, depth=0):
        # Determine the number of samples and features in the input data, and the number of unique labels
        n_samples, features = X.shape
        n_labels = len (np.unique (y))

        # If the maximum depth has been reached, or there is only one unique label left in the data,
        # or there are fewer samples than the minimum required to split a node, return a leaf node
        if depth >= self.max_depth or n_labels == 1 or n_samples < self.min_samples_split:
            leaf_value = self.most_common_label (y)
            return DecisionNode (value=leaf_value)

        # Randomly select a subset of features to consider at each split
        feature_indexes = np.random.choice (features, self.n_features, replace=False)

        # Determine the best feature and threshold to split the data based on the selected features
        best_feature, best_thresh = self.best_split (X, y, feature_indexes)

        # Split the data into two subsets based on the best feature and threshold, and recursively build the
        # left and right subtrees using these subsets
        left_indexes, right_indexes = self.split (X[:, best_feature], best_thresh)
        left = self.build_tree (X[left_indexes, :], y[left_indexes], depth + 1)
        right = self.build_tree (X[right_indexes, :], y[right_indexes], depth + 1)

        # Return a DecisionNode object representing the current split, with pointers to the left and right subtrees
        return DecisionNode (best_feature, best_thresh, left, right)

    def best_split(self, X, y, feat_indexes):
        # Initialize variables to keep track of the best feature, threshold, and information gain
        best_gain = -1
        split_idx, split_threshold = None, None

        # Iterate over each feature and each unique value of that feature to determine the feature and threshold
        # that maximizes the information gain
        for feat_idx in feat_indexes:
            X_column = X[:, feat_idx]
            thresholds = np.unique (X_column)

            for n in thresholds:
                gain = self.information_gain (y, X_column, n)

                if gain > best_gain:
                    best_gain = gain
                    split_idx = feat_idx
                    split_threshold = n

        # Return the best feature and threshold
        return split_idx, split_threshold

    def information_gain(self, y, X_column, threshold):
        # calculate parent entropy
        parent_entropy = self.entropy (y)

        # split the data based on the given threshold
        left_indexes, right_indexes = self.split (X_column, threshold)

        # if either left or right set is empty, return 0
        if len (left_indexes) == 0 or len (right_indexes) == 0:
            return 0

        # calculate child entropy for left and right sets
        n = len (y)
        n_left, n_right = len (left_indexes), len (right_indexes)
        entropy_left, entropy_right = self.entropy (y[left_indexes]), self.entropy (y[right_indexes])
        child_entropy = (n_left / n) * entropy_left + (n_right / n) * entropy_right

        # calculate information gain
        information_gain = parent_entropy - child_entropy
        return information_gain

    def split(self, data_index, threshold):
        # split data based on the given threshold
        left = np.argwhere (data_index <= threshold).flatten ()
        right = np.argwhere (data_index > threshold).flatten ()
        return left, right

    def entropy(self, y):
        # calculate entropy of a given set
        hist = np.bincount (y)
        ps = hist / len (y)
        return -np.sum ([p * np.log (p) for p in ps if p > 0])

    def most_common_label(self, y):
        # return the most common label in a given set
        counter = Counter (y)
        value = counter.most_common (1)[0][0]
        return value

    def predict(self, X):
        # predict the output for a given set of inputs
        return np.array ([self.traverse_tree (x, self.root) for x in X])

    def traverse_tree(self, x, node):
        # traverse the decision tree recursively to get the output for a given input
        if node.is_leaf_node ():
            return node.value

        if x[node.feature] <= node.threshold:
            return self.traverse_tree (x, node.left)
        return self.traverse_tree (x, node.right)
