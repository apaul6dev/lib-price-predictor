import unittest
from interface.predictor import test_model
from sklearn.datasets import make_regression
from core.evaluator import evaluate

class ModelSelectionTest(unittest.TestCase):
    def test_dispatch_random_forest(self):
        X, y = make_regression(n_samples=100, n_features=5, noise=0.1)
        preds = test_model("random_forest", X[:70], y[:70], X[70:], y[70:])
        self.assertEqual(len(preds), 30)

if __name__ == "__main__":
    unittest.main()
