import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from itertools import cycle
from plotnine import ggplot, aes, geom_line, labs, theme_bw, scale_color_manual, ggtitle
from sklearn_ml.ensemble_classifier import EnsembleClassifier
from sklearn_ml.pipeline_utils import PipelineBuilder

class TestPipelineAndEnsemble(unittest.TestCase):

    def setUp(self):
        # Create synthetic dataset
        n_samples = 1000
        X, y = make_classification(n_samples=n_samples, n_features=20, n_informative=15, n_redundant=5, random_state=42)
        weights = np.ones_like(y, dtype=float)
        X_all = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(X.shape[1])])
        nn = n_samples*80//100
        X_train_w_cols = X_all.iloc[:nn]
        y_train = pd.Series(y[:nn])
        wt_train = pd.Series(weights[:nn])
        X_test_w_cols = X_all.iloc[nn:]
        y_test = pd.Series(y[nn:])
        wt_test = pd.Series(weights[nn:])

        self.model_data = (X_all, X_train_w_cols, y_train, wt_train, X_test_w_cols, y_test, wt_test)
        self.excl_model_cols = []
        self.bench_scores = []

    def test_pipeline_building(self):
        pipeline_lin, pipeline_nlin = PipelineBuilder.build_pipeline(self.model_data[0], self.excl_model_cols)
        self.assertIsNotNone(pipeline_lin, "Linear pipeline should not be None")
        self.assertIsNotNone(pipeline_nlin, "Non-linear pipeline should not be None")

    def test_ensemble_classifier_run(self):
        ensemble = EnsembleClassifier(self.model_data, self.excl_model_cols, self.bench_scores)
        result = ensemble.run(include_stacking=True, perform_cv=False)
        self.assertIsInstance(result, pd.DataFrame, "The result should be a DataFrame")
        self.assertFalse(result.empty, "The result DataFrame should not be empty")

    def test_classifier_estimators_building(self):
        pipeline_lin, pipeline_nlin = PipelineBuilder.build_pipeline(self.model_data[0], self.excl_model_cols)
        ensemble = EnsembleClassifier(self.model_data, self.excl_model_cols, self.bench_scores)
        estimators, stacking_clf = ensemble.build_classifier_estimators(pipeline_lin, pipeline_nlin)
        self.assertTrue(len(estimators) > 0, "There should be at least one estimator")
        self.assertIsNotNone(stacking_clf, "Stacking classifier should not be None")


if __name__ == '__main__':
    unittest.main()
