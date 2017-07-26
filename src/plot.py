import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import numpy as np

import src.perceptron

train_data = [[1.0, 2.0, 4.0, 3.0], [2.0, 3.0, 3.0, 5.0]]
target_values = [1, 1, -1, -1]
weight = [4.0, -1.5, -1.0]
rate = 0.1

perceptron = src.perceptron.Perceptron(weight, rate, train_data, target_values)
weights_all = perceptron.train()

line1, = plt.plot(train_data[0][:2], train_data[1][:2], '^', label='Class 1')
plt.plot(train_data[0][2:], train_data[1][2:], 'o', label='Class 2')

t = np.arange(0.0, 6.0, 0.01)
for index, value in enumerate(weights_all):
    y = (-t * value[1] - value[0]) / value[2]
    line, = plt.plot(t, y, label="w_" + str(index))
    plt.legend(handler_map={line: HandlerLine2D(numpoints=4)})

axes = plt.gca()
axes.set_xlim([0.0, 6.0])
axes.set_ylim([0.0, 6.0])

plt.legend(handler_map={line1: HandlerLine2D(numpoints=4)})

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('Train data with all weights of perceptron algorithm')
plt.grid(True)
plt.savefig("plot.png")
plt.show()
