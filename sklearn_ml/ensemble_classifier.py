'''
Ensemble binary classifier based on sklearn.
'''
from functools import wraps
import copy
import time
import logging
import pandas as pd
import numpy as np

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict
from sklearn.ensemble import (RandomForestClassifier,
                              HistGradientBoostingClassifier,
                              AdaBoostClassifier,
                              BaggingClassifier,
                              StackingClassifier)
from sklearn.linear_model import LogisticRegression
from lightgbm import LGBMClassifier

from .pipeline_utils import PipelineBuilder

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
    ''' wrapper class for ensemble classifier
    '''

    def __init__(self, model_data, excl_model_cols=None, bench_scores=None,
                 numeric_types=(int, np.int16, np.int32, np.int64,
                                float, np.float16, np.float32, np.float64,
                                np.double)):
        ''' note x_train_w_cols contain some score columns for benchmarking purpose
            they should not be part of model building
        '''
        if excl_model_cols is None:
            excl_model_cols = []
        if bench_scores is None:
            bench_scores = []

        self.x_all, self.x_train_w_cols, self.y_train, self.wt_train, self.x_test_w_cols, self.y_test, self.wt_test = model_data
        self.excl_model_cols = excl_model_cols
        self.bench_scores = bench_scores

        for scr in bench_scores:
            if not isinstance(self.x_all[scr][0], numeric_types):
                raise ValueError(f'ERROR: bench score {scr} column is not numeric type')

        self.features = None
        self.res = None
        self.estimators = None

    def run(self, include_stacking=False, perform_cv=True, cv_folds=5, n_jobs=None, **kwargs):
        '''
           step 1: if perform_cv=True, cross_val_predict() is called on training dataset
                   and corresponding cross validated performances are produced;
                   Note: this step is the most time consuming (4x) compared to step 2
           step 2: training dataset is used to fit individual classifier and stacking classifier
           step 3: predict_proba() is called to produce probability for test dataset
        '''
        p_lin, p_nlin = PipelineBuilder.build_pipeline(self.x_all, self.excl_model_cols)
        clfs, stacking_clf = self.build_classifier_estimators(p_lin, p_nlin, **kwargs)

        if not include_stacking:
            estimators = clfs
        else:
            estimators = clfs + [('StackingClassifier', stacking_clf)]

        res = pd.DataFrame({})
        features = pd.DataFrame({})

        x_train_w_cols, x_test_w_cols = self.x_train_w_cols, self.x_test_w_cols
        x_train = x_train_w_cols.drop(columns=self.excl_model_cols)
        y_train, wt_train = self.y_train, self.wt_train

        x_test = x_test_w_cols.drop(columns=self.excl_model_cols)
        y_test, wt_test = self.y_test, self.wt_test

        for (name, pipe_est) in estimators:
            if name != 'StackingClassifier':
                est_name = str(pipe_est[1]).split('(', maxsplit=1)[0].lower()
                wt_train_copy = copy.copy(wt_train)
                if perform_cv:
                    pred_cv = cross_val_predict(pipe_est, x_train, y_train,
                                                fit_params={f'{est_name}__sample_weight': wt_train_copy},
                                                method='predict_proba',
                                                cv=cv_folds,
                                                n_jobs=n_jobs)
                start = time.time()
                pipe_est.fit(x_train, y_train, **{f'{est_name}__sample_weight': wt_train_copy})
                end = time.time()
                logger.info(f'{name} fit timing: {end-start:.4f}')
            else:
                if perform_cv:
                    pred_cv = cross_val_predict(pipe_est, x_train, y_train, method='predict_proba', cv=cv_folds, n_jobs=n_jobs)
                start = time.time()
                pipe_est.fit(x_train, y_train)
                end = time.time()
                logger.info(f'{name} fit timing: {end-start:.4f}')

            if hasattr(pipe_est, 'steps') and hasattr(pipe_est.steps[1][1], 'feature_importances_'):
                importances = pipe_est.steps[1][1].feature_importances_
            elif name == 'Bagging':
                importances = np.mean([v.feature_importances_ for v in pipe_est[1].estimators_], axis=0)
            else:
                importances = []

            if len(importances) > 0:
                features0 = pd.DataFrame({'model': name, 'variable': x_train.columns,
                                          'importance': importances}).sort_values('importance', ascending=False)
                features0['rank'] = 1
                features0['rank'] = features0['rank'].cumsum()
                features = pd.concat([features, features0], axis=0)

            pred0 = pipe_est.predict_proba(x_test)
            res_test = pd.DataFrame({'model': name, 'type': 'test', 'actual': self.y_test, 'pred': pred0[:, 1], 'wt': wt_test})

            if perform_cv:
                res_cv = pd.DataFrame({'model': name, 'type': 'cv', 'actual': self.y_train, 'pred': pred_cv[:, 1], 'wt': wt_train})
                res = pd.concat([res, res_cv, res_test], axis=0)
            else:
                res = pd.concat([res, res_test], axis=0)

        for scr in self.bench_scores:
            if x_train_w_cols[scr].isna().all() or x_train_w_cols[scr].max() == 0:
                logger.warning(f'Benchmark score column {scr} contains all NaNs or has a max value of 0, skipping this column.')
                continue

            y_pred = 1 - x_train_w_cols[scr].fillna(0)/x_train_w_cols[scr].fillna(0).max()
            df_y = pd.concat([y_train, y_pred, wt_train], axis=1)
            df_y.columns = ['actual', 'pred', 'wt']
            df_y['model'] = scr
            df_y['type'] = 'cv'

            cols_output = ['model', 'type', 'actual', 'pred', 'wt']
            res = pd.concat([res, df_y[cols_output]], axis=0)

            if x_test_w_cols[scr].isna().all() or x_test_w_cols[scr].max() == 0:
                logger.warning(f'Benchmark score column {scr} in test set contains all NaNs or has a max value of 0, skipping this column.')
                continue

            y_pred = 1 - x_test_w_cols[scr].fillna(0)/x_test_w_cols[scr].fillna(0).max()

            df_test = pd.concat([y_test, y_pred, wt_test], axis=1)
            df_test.columns = ['actual', 'pred', 'wt']
            df_test['model'] = scr
            df_test['type'] = 'test'

            res = pd.concat([res, df_test[cols_output]], axis=0)

        self.features = features
        self.res = res
        self.estimators = estimators

        return res

    def build_classifier_estimators(self, processor_lin, processor_nlin, list_of_estimators=None, final_estimator='RandomForest'):
        ''' build pipelines:
              1. missing data imputation based on data type;
              2. scale data for linear estimator (LogisticRegression)
              3. preprocess all columns based on data type based on 1 & 2
              4. create pipeline for each estimator defined by user
        '''
        if list_of_estimators is None:
            list_of_estimators = []

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

        all_est = dict(linear_est + nonlinear_est)

        print(f'INFO: building classfier based on {list_of_estimators = }')

        estimators = []
        for name, est in linear_est:
            if len(list_of_estimators) == 0 or name in list_of_estimators:
                estimators.append((name, make_pipeline(processor_lin, est)))

        for name, est in nonlinear_est:
            if len(list_of_estimators) == 0 or name in list_of_estimators:
                estimators.append((name, make_pipeline(processor_nlin, est)))

        if final_estimator in all_est:
            fin_est = all_est[final_estimator]
            print(f'INFO: ensemble estimator {final_estimator}')
        else:
            raise ValueError(f'ERROR: {final_estimator} not in {all_est = }')

        stacking_classifier = StackingClassifier(estimators=estimators,
                                                 final_estimator=fin_est,
                                                 stack_method='predict_proba')

        return estimators, stacking_classifier
