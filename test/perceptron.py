import unittest

import src.perceptron

class TestPerceptorn(unittest.TestCase):

    def first_test_case(self):
        self.assertEqual(src.perceptron.test_method(), 2)