import unittest

import src.perceptron


class TestPerceptorn(unittest.TestCase):

    def test_init(self):
        train_data = [[1.0, 2.0, 4.0, 3.0], [2.0, 3.0, 3.0, 5.0]]
        target_values = [1, 1, -1, -1]
        weight = [4.0, -1.5, -1.0]
        rate = 0.1;

        perceptron = src.perceptron.Perceptron(weight, rate, train_data, target_values)
        self.assertEqual(perceptron.weight, weight)
        self.assertEqual(perceptron.rate, rate)
        self.assertEqual(perceptron.train_data, train_data)
        self.assertEqual(perceptron.target_values, target_values)


    def test_train(self):
        train_data = [[1.0, 2.0, 4.0, 3.0], [2.0, 3.0, 3.0, 5.0]]
        target_values = [1, 1, -1, -1]
        weight = [4.0, -1.5, -1.0]
        rate = 0.1

        perceptron = src.perceptron.Perceptron(weight, rate, train_data, target_values)

        weights_all = perceptron.train()
        result = weights_all.pop()
        
        for i in range(0, 3, 1):
            result[i] = round(result[i], 1)
        self.assertEqual(result, [4.2, -1.1, -0.4])

    def test_train_number_of_iteration(self):
        train_data = [[1.0, 2.0, 4.0, 3.0], [2.0, 3.0, 3.0, 5.0]]
        target_values = [1, 1, -1, -1]
        weight_start = [4.0, -1.5, -1.0]
        rate = 0.1

        perceptron = src.perceptron.Perceptron(weight_start, rate, train_data, target_values)

        weights_all = perceptron.train()
        self.assertEqual(len(weights_all), 3)

    def test_classify(self):
        weight = [4.1, -1.3, -0.7]
        train_data = [[1.0, 2.0, 4.0, 3.0], [2.0, 3.0, 3.0, 5.0]]
        target_values = [1, 1, -1, -1]
        rate = 0.1;

        perceptron = src.perceptron.Perceptron(weight, rate, train_data, target_values)
        result = perceptron.classify(weight, [3.0, 4.0])
        self.assertEqual(result, -1)

    def test_linear_separation(self):
        weight = [4.2, -1.1, -0.4]
        train_data = [[1.0, 2.0, 4.0, 3.0], [2.0, 3.0, 3.0, 5.0]]
        target_values = [1, 1, -1, -1]
        rate = 0.1;

        perceptron = src.perceptron.Perceptron(weight, rate, train_data, target_values)
        result = perceptron.is_linear_separation(weight)
        self.assertTrue(result[0])
