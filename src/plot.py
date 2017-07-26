import matplotlib.pyplot as plt
import numpy as np

import src.perceptron

train_data = [[1.0, 2.0, 4.0, 3.0], [2.0, 3.0, 3.0, 5.0]]
target_values = [1, 1, -1, -1]
weight = [4.0, -1.5, -1.0]
rate = 0.1;

perceptron = src.perceptron.Perceptron(weight, rate, train_data, target_values)
weights_all = perceptron.train()

t = np.arange(0.0, 6.0, 0.01)

for i in weights_all:
    y = (-t * i[1] - i[0]) / i[2]
    plt.plot(t, y)

plt.plot(train_data[0][:2], train_data[1][:2], '^')
plt.plot(train_data[0][2:], train_data[1][2:], 'o')

axes = plt.gca()
axes.set_xlim([0.0, 6.0])
axes.set_ylim([0.0, 6.0])

plt.xlabel('x_1')
plt.ylabel('x_2')
plt.title('Train data with all weights of perceptron algorithm')
plt.grid(True)
plt.savefig("plot.png")
plt.show()
