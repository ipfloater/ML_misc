import unittest
import numpy as np
import pandas as pd
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import matplotlib.pyplot as plt
from itertools import cycle
from plotnine import ggplot, aes, geom_line, labs, theme_bw, scale_color_manual, ggtitle
from sklearn_ML.ensemble_classifier import EnsembleClassifier

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

    def test_roc_auc_and_precision_recall_curve(self):
        # Run the ensemble classifier
        ensemble = EnsembleClassifier(self.model_data, self.excl_model_cols, self.bench_scores)
        result = ensemble.run(include_stacking=True, perform_cv=False)

        # Filter results for test set predictions
        test_results = result[result['type'] == 'test']

        # Colors for plotting
        colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])

        # Plot ROC AUC curve for each model using matplotlib
        plt.figure()
        for model_name in test_results['model'].unique():
            model_results = test_results[test_results['model'] == model_name]
            y_test = model_results['actual']
            y_pred = model_results['pred']
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_auc = auc(fpr, tpr)
            color = next(colors)
            plt.plot(fpr, tpr, color=color, lw=2, label=f'{model_name} (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve by Model')
        plt.legend(loc='lower right')
        plt.show()

        # Plot Precision-Recall curve for each model using matplotlib
        plt.figure()
        colors = cycle(['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])
        for model_name in test_results['model'].unique():
            model_results = test_results[test_results['model'] == model_name]
            y_test = model_results['actual']
            y_pred = model_results['pred']
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            color = next(colors)
            plt.plot(recall, precision, color=color, lw=2, label=f'{model_name}')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve by Model')
        plt.legend(loc='lower left')
        plt.show()

    def test_roc_auc_and_precision_recall_curve_plotnine(self):
        # Run the ensemble classifier
        ensemble = EnsembleClassifier(self.model_data, self.excl_model_cols, self.bench_scores)
        result = ensemble.run(include_stacking=True, perform_cv=False)

        # Filter results for test set predictions
        test_results = result[result['type'] == 'test']

        # Prepare data for ROC AUC curve with plotnine
        roc_data = []
        for model_name in test_results['model'].unique():
            model_results = test_results[test_results['model'] == model_name]
            y_test = model_results['actual']
            y_pred = model_results['pred']
            fpr, tpr, _ = roc_curve(y_test, y_pred)
            roc_data.append(pd.DataFrame({
                'fpr': fpr,
                'tpr': tpr,
                'model': model_name
            }))
        roc_df = pd.concat(roc_data, axis=0)

        # Plot ROC AUC curve using plotnine
        roc_plot = (
            ggplot(roc_df, aes(x='fpr', y='tpr', color='model')) +
            geom_line() +
            labs(x='False Positive Rate', y='True Positive Rate', title='Receiver Operating Characteristic (ROC) Curve by Model') +
            theme_bw() +
            scale_color_manual(values=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])
        )
        print(roc_plot)

        # Prepare data for Precision-Recall curve with plotnine
        pr_data = []
        for model_name in test_results['model'].unique():
            model_results = test_results[test_results['model'] == model_name]
            y_test = model_results['actual']
            y_pred = model_results['pred']
            precision, recall, _ = precision_recall_curve(y_test, y_pred)
            pr_data.append(pd.DataFrame({
                'recall': recall,
                'precision': precision,
                'model': model_name
            }))
        pr_df = pd.concat(pr_data, axis=0)

        # Plot Precision-Recall curve using plotnine
        pr_plot = (
            ggplot(pr_df, aes(x='recall', y='precision', color='model')) +
            geom_line() +
            labs(x='Recall', y='Precision', title='Precision-Recall Curve by Model') +
            theme_bw() +
            scale_color_manual(values=['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black'])
        )
        print(pr_plot)


if __name__ == '__main__':
    unittest.main()
