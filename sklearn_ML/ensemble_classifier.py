from functools import reduce, wraps
import copy
import pandas as pd
import numpy as np
import time
import logging

from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import (RandomForestClassifier, HistGradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, StackingClassifier)
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from .pipeline_utils import *

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def timethis(func):
    '''
    Decorator that reports execution time
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(f'{func.__name__} timing: {end-start:.4f}')
        return result
    return wrapper


cross_val_predict = timethis(cross_val_predict)

class EnsembleClassifier:
    def __init__(self, model_data, excl_model_cols=[], bench_scores=[],
                 numeric_types=(int, np.int16, np.int32, np.int64,
                                float, np.float16, np.float32, np.float64,
                                np.double)):
        ''' note X_train_w_cols contain some score columns for benchmarking purpose
            they should not be part of model building
        '''
        self.X_all, self.X_train_w_cols, self.y_train, self.wt_train, self.X_test_w_cols, self.y_test, self.wt_test = model_data
        self.excl_model_cols = excl_model_cols
        self.bench_scores = bench_scores

        for scr in bench_scores:
            if not isinstance(self.X_all[scr][0], numeric_types):
                raise ValueError(f'ERROR: bench score {scr} column is not numeric type')

    def run(self, include_stacking=False, perform_cv=True, cv_folds=5, n_jobs=None, **kwargs):
        '''
           step 1: if perform_cv=True, cross_val_predict() is called on training dataset
                   and corresponding cross validated performances are produced;
                   Note: this step is the most time consuming (4x) compared to step 2
           step 2: training dataset is used to fit individual classifier and stacking classifier
           step 3: predict_proba() is called to produce probability for test dataset
        '''
        p_lin, p_nlin = PipelineBuilder.build_pipeline(self.X_all, self.excl_model_cols)
        clfs, stacking_clf = self.build_classifier_estimators(p_lin, p_nlin)

        if not include_stacking:
            estimators = clfs
        else:
            estimators = clfs + [('StackingClassifier', stacking_clf)]

        res = pd.DataFrame({})
        features = pd.DataFrame({})

        X_train_w_cols, X_test_w_cols = self.X_train_w_cols, self.X_test_w_cols
        X_train = X_train_w_cols.drop(columns=self.excl_model_cols)
        y_train, wt_train = self.y_train, self.wt_train

        X_test = X_test_w_cols.drop(columns=self.excl_model_cols)
        y_test, wt_test = self.y_test, self.wt_test

        for (name, est) in estimators:
            if name != 'StackingClassifier':
                nm = str(est[1]).split('(')[0].lower()
                wt_train_copy = copy.copy(wt_train)
                if perform_cv:
                    pred_cv = cross_val_predict(est, X_train, y_train, fit_params={
                                                f'{nm}__sample_weight': wt_train_copy}, method='predict_proba', cv=cv_folds, n_jobs=n_jobs)
                start = time.time()
                est.fit(X_train, y_train, **{f'{nm}__sample_weight': wt_train_copy})
                end = time.time()
                logger.info(f'{name} fit timing: {end-start:.4f}')
            else:
                if perform_cv:
                    pred_cv = cross_val_predict(est, X_train, y_train, method='predict_proba', cv=cv_folds, n_jobs=n_jobs)
                start = time.time()
                est.fit(X_train, y_train)
                end = time.time()
                logger.info(f'{name} fit timing: {end-start:.4f}')

            if hasattr(est, 'steps') and hasattr(est.steps[1][1], 'feature_importances_'):
                importances = est.steps[1][1].feature_importances_
            elif name == 'Bagging':
                importances = np.mean([v.feature_importances_ for v in est[1].estimators_], axis=0)
            else:
                importances = []

            if len(importances) > 0:
                features0 = pd.DataFrame({'model': name, 'variable': X_train.columns,
                                          'importance': importances}).sort_values('importance', ascending=False)
                features0['rank'] = 1
                features0['rank'] = features0['rank'].cumsum()
                features = pd.concat([features, features0], axis=0)

            pred0 = est.predict_proba(X_test)
            res_test = pd.DataFrame({'model': name, 'type': 'test', 'actual': self.y_test, 'pred': pred0[:, 1], 'wt': wt_test})

            if perform_cv:
                res_cv = pd.DataFrame({'model': name, 'type': 'cv', 'actual': self.y_train, 'pred': pred_cv[:, 1], 'wt': wt_train})
                res = pd.concat([res, res_cv, res_test], axis=0)
            else:
                res = pd.concat([res, res_test], axis=0)

        for scr in self.bench_scores:
            if X_train_w_cols[scr].isna().all() or X_train_w_cols[scr].max() == 0:
                logger.warning(f'Benchmark score column {scr} contains all NaNs or has a max value of 0, skipping this column.')
                continue

            y_pred = 1 - X_train_w_cols[scr].fillna(0)/X_train_w_cols[scr].fillna(0).max()
            a = pd.concat([y_train, y_pred, wt_train], axis=1)
            a.columns = ['actual', 'pred', 'wt']
            a['model'] = scr
            a['type'] = 'cv'

            res = pd.concat([res, a[['model', 'type', 'actual', 'pred', 'wt']]], axis=0)

            if X_test_w_cols[scr].isna().all() or X_test_w_cols[scr].max() == 0:
                logger.warning(f'Benchmark score column {scr} in test set contains all NaNs or has a max value of 0, skipping this column.')
                continue

            y_pred = 1 - X_test_w_cols[scr].fillna(0)/X_test_w_cols[scr].fillna(0).max()

            a = pd.concat([y_test, y_pred, wt_test], axis=1)
            a.columns = ['actual', 'pred', 'wt']
            a['model'] = scr
            a['type'] = 'test'

            res = pd.concat([res, a[['model', 'type', 'actual', 'pred', 'wt']]], axis=0)

        self.features = features
        self.res = res
        self.estimators = estimators

        return res

    def build_classifier_estimators(self, processor_lin, processor_nlin, list_of_estimators=[]):
        linear_est = [
            ('Logi', LogisticRegression(max_iter=10000)),
        ]

        nonlinear_est = [
            ('LGBM', LGBMClassifier()),
            ('RandomForest', RandomForestClassifier(random_state=42)),
            ('HistGradientBoosting', HistGradientBoostingClassifier(random_state=0)),
            ('AdaBoosting', AdaBoostClassifier(learning_rate=0.5, n_estimators=100)),
            ('Bagging', BaggingClassifier(n_estimators=10)),
        ]

        estimators = []
        for name, est in linear_est:
            if len(list_of_estimators) == 0 or name in list_of_estimators:
                estimators.append((name, make_pipeline(processor_lin, est)))

        for name, est in nonlinear_est:
            if len(list_of_estimators) == 0 or name in list_of_estimators:
                estimators.append((name, make_pipeline(processor_nlin, est)))

        stacking_classifier = StackingClassifier(estimators=estimators,
                                                 final_estimator=LGBMClassifier(class_weight='balanced'),
                                                 stack_method='predict_proba')

        return estimators, stacking_classifier
