import unittest

from src.utils.metrics import evaluate


class ThresholdTests(unittest.TestCase):
    def test_probability_threshold_outside_quantile_grid(self):
        metrics = evaluate([1, 1, 0], [0.7, 0.8, 0.9])
        self.assertEqual(metrics["f1"], 0.8)
        self.assertEqual(metrics["f1_tuned_thr"], 0.8)

    def test_svc_threshold_outside_quantile_grid(self):
        metrics = evaluate([1, 1, 0], [0.1, 0.2, 0.3], threshold=0.0)
        self.assertEqual(metrics["f1"], 0.8)
        self.assertEqual(metrics["f1_tuned_thr"], 0.8)


if __name__ == "__main__":
    unittest.main()
