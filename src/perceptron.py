import numpy as np
from numpy import linalg as LA

"""
This class implements a linear classifier with the perceptron algorithm for a initial weight, train and target vector.
"""


class Perceptron():
    weights = []
    THRESHOLD = 0.001

    def __init__(self, weight, rate, train_data, target_values):
        self.weights.append(weight)
        self.rate = rate
        self.train_data = train_data
        self.target_values = target_values

    # First calculate all feature vectors, which are not correct classified with the current weight. Afterwards apply
    # the perceptron learning rule to them.
    def train(self):
        weight_old = self.weights[len(self.weights) - 1]
        (is_lin_sep, y_w) = self.is_linear_separation(weight_old)
        delta = 0

        while is_lin_sep == False or delta > self.THRESHOLD:
            sum = [0.0, 0.0, 0.0]
            for i in y_w:
                sum[0] = sum[0] - 1
                sum[1] = sum[1] - self.target_values[i] * self.train_data[0][i]
                sum[2] = sum[2] - self.target_values[i] * self.train_data[1][i]

            weight_new = [0.0, 0.0, 0.0]
            for i in range(0, 3, 1):
                weight_new[i] = weight_old[i] - self.rate * sum[i]

            self.weights.append(weight_new)
            delta = LA.norm(np.subtract(weight_old, weight_new))
            weight_old = self.weights[len(self.weights) - 1]
            (is_lin_sep, y_w) = self.is_linear_separation(weight_old)

        return self.weights

    # Method determines whether the current weight classifies a feature vector as member of class 1 or -1.
    def classify(self, weight, x_vec):
        value = weight[0]
        for i in range(1, 3, 1):
            value = value + weight[i] * x_vec[i - 1]
        if value > 0:
            return 1
        return -1

    # Method checks if all of the train data is separated into two disjoint classes by the current weight vector.
    def is_linear_separation(self, weight):

        # Stores all not correct classified feature vectors
        y_w = []
        for i in range(0, len(self.train_data[0]), 1):
            x_1 = self.train_data[0][i]
            x_2 = self.train_data[1][i]
            if self.classify(weight, [x_1, x_2]) != self.target_values[i]:
                y_w.append(i)

        if y_w == []:
            return (True, y_w)
        else:
            return (False, y_w)
