import copy

from sklearn.pipeline import make_pipeline
from sklearn.model_selection import cross_val_predict

from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, BaggingClassifier
from sklearn.ensemble import StackingRegressor
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LassoCV
from sklearn.linear_model import RidgeCV

from sklearn.experimental import enable_hist_gradient_boosting  # noqa
from sklearn.ensemble import HistGradientBoostingClassifier

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.linear_model import RidgeClassifierCV, RidgeClassifier

from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier

from lightgbm import LGBMClassifier

from .utils import *

cross_val_predict = timethis(cross_val_predict)

class classifier():
    def __init__(self, model_data, excl_model_cols=[], bench_scores=[],
                 numeric_types=(int, np.int16, np.int32, np.int64,
                                float, np.float16, np.float32, np.float64,
                                np.double)):
        ''' note X_train_w_cols contain some socre columns for benchmarking purpose
            they should not be part of model building
        '''
        self.X_all, self.X_train_w_cols, self.y_train, self.wt_train, self.X_test_w_cols, self.y_test, self.wt_test = model_data
        self.excl_model_cols = excl_model_cols
        self.bench_scores = bench_scores

        for scr in bench_scores:
            if not isinstance(self.X_all[scr][0], numeric_types):
                raise ValueError(f'ERROR: bench score {scr} column is not numeric type')

    def run(self, include_stacking=False, perform_cv=True, **kwargs):
        '''
           step 1: if perform_cv=True, cross_val_predict() is called on training dataset
                   and corresponding cross validated performances are produced;
                   Note: this step is the most time consuming (4x) compared to step 2
           step 2: training dataset is used to fit individual classifier and stacking classifier
           step 3: predict_proba() is called to produce probability for test dataset
        '''
        p_lin, p_nlin = self.build_pipeline(self.X_all, self.excl_model_cols)
        # import pdb; pdb.set_trace()
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

        # print(f'pre-loop wt_train {wt_train.shape} {np.mean(wt_train):.3f}  wt_test {wt_test.shape} {np.mean(wt_test):.3f}')

        for (name, est) in estimators:
            if name != 'StackingClassifier':
                nm = str(est[1]).split('(')[0].lower()
                print('=========================================')
                print(name, nm)
                # AdaBoostClassifier.fit() will normalize sample_weight
                wt_train_copy = copy.copy(wt_train)
                if perform_cv:
                    # pass "sample_weight=" keyword through pipeline
                    pred_cv = cross_val_predict(est, X_train, y_train, fit_params={f'{nm}__sample_weight': wt_train_copy}, method='predict_proba')
                start = time.time()
                est.fit(X_train, y_train, **{f'{nm}__sample_weight': wt_train_copy})
                end = time.time()
                print(f'{name} fit timing: {end-start:.4f}')
            else:
                # can't pass sample_weights to piped estimators
                # run equal-weighted instead
                print(name)
                if perform_cv:
                    pred_cv = cross_val_predict(est, X_train, y_train, method='predict_proba')
                start = time.time()
                est.fit(X_train, y_train)
                end = time.time()
                print(f'{name} fit timing: {end-start:.4f}')

            if 'steps' in dir(est) and 'feature_importances_' in dir(est.steps[1][1]):
                importances = est.steps[1][1].feature_importances_
            elif name == 'Bagging':
                importances = np.mean(np.array([v.feature_importances_ for v in est[1].estimators_]), axis=0)
            else:
                print(f'{name} does not have feature importances')
                importances = []

            if len(importances) > 0:
                features0 = pd.DataFrame({'model': name, 'variable': X_train.columns,
                                          'importance': importances}).sort_values('importance', ascending=False)
                features0['rank'] = 1
                features0['rank'] = features0['rank'].cumsum()
                print(features0.head(5))
                features = pd.concat([features, features0], axis=0)

            # this is slow for BaggingClassifier
            pred0 = est.predict_proba(X_test)
            res_test = pd.DataFrame({'model': name, 'type': 'test', 'actual': self.y_test, 'pred': pred0[:, 1], 'wt': wt_test})

            if perform_cv:
                res_cv = pd.DataFrame({'model': name, 'type': 'cv', 'actual': self.y_train, 'pred': pred_cv[:, 1], 'wt': wt_train})
                res = pd.concat([res, res_cv, res_test], axis=0)
                print(f'pred_cv {pred_cv.shape}', end=' ')
            else:
                res = pd.concat([res, res_test], axis=0)

            print(f'wt_train {wt_train.shape} {np.mean(wt_train):.3f} pred0 {pred0.shape} wt_test {wt_test.shape} {np.mean(wt_test):.3f}')

        for scr in self.bench_scores:
            y_pred = 1 - X_train_w_cols[scr].fillna(0)/X_train_w_cols[scr].fillna(0).max()
            a = pd.concat([y_train, y_pred, wt_train], axis=1)
            a.columns = ['actual', 'pred', 'wt']
            a['model'] = scr
            a['type'] = 'cv'

            res = pd.concat([res, a[['model', 'type', 'actual', 'pred', 'wt']]], axis=0)

            y_pred = 1 - X_test_w_cols[scr].fillna(0)/X_test_w_cols[scr].fillna(0).max()

            a = pd.concat([y_test, y_pred, wt_test], axis=1)
            a.columns = ['actual', 'pred', 'wt']
            a['model'] = scr
            a['type'] = 'test'

            res = pd.concat([res, a[['model', 'type', 'actual', 'pred', 'wt']]], axis=0)
            # print(res)

        self.features = features
        self.res = res
        self.estimators = estimators

        return res

    def gen_perf_stats(self, **kwargs):
        stats = []
        popCaps, binCaps = pd.DataFrame(), pd.DataFrame()

        res = self.res
        verb = kwargs['verb'] if 'verb' in kwargs.keys() else False

        for mdl in res['model'].unique():
            for type in ['cv', 'test']:
                gg = res.query(f'model=="{mdl}" and type=="{type}"')
                if gg.shape[0] == 0:
                    continue
                print(mdl, type)
                auc0, ks0, capt5, capt10, capt20, count, wtCount, popCap, rocCurve, binCap = gen_cap_curve(
                    gg['actual'], gg['pred'], gg['wt'], mdl, type, '', verb)
                pf = (mdl, type, ks0, 2*auc0-1, auc0, capt5, capt10, capt20, count, wtCount)
                stats += [pf]
                popCaps = pd.concat([popCaps, popCap], axis=0)
                binCaps = pd.concat([binCaps, binCap], axis=0)

        stats = pd.DataFrame(stats, columns=['model', 'type', 'ks', 'gini', 'auc', 'cap5%', 'cap10%', 'cap20%', 'count', 'wtCount'])
        stats = stats.sort_values('type')

        return stats, popCaps, binCaps

    @staticmethod
    def build_pipeline(XX_all, excl_model_cols=[]):
        ''' XX_all should contain all possible values for categoricals,
            otherwise pipeline will complain in processing X_test
        '''
        def extract_col_types(XX, missing_cat):
            ''' missing_cat is the imputing string for categorical NA
            '''
            cat_cols = XX.columns[XX.dtypes == 'O']
            num_cols = XX.columns[XX.dtypes != 'O']

            # OneHotEncoder applied to linear estimator only!
            categories = [[v if v is not None else missing_cat for v in XX[column].unique()] for column in XX[cat_cols]]
            return num_cols, cat_cols, categories

        # cat_cols = XX_num.columns[XX_num.dtypes == 'O']
        # num_cols = XX_num.columns[XX_num.dtypes != 'O']
        # ## OneHotEncoder applied to linear estimator only!
        # categories = [[v if v is not None else missing_cat for v in XX_num[column].unique()] for column in XX_num[cat_cols]]

        missing_cat = 'missing'

        # make sure to use entire sample to build categories for categorical variables!
        num_cols, cat_cols, categories = extract_col_types(XX_all.drop(columns=excl_model_cols), missing_cat)

        # numeric
        num_proc_lin = make_pipeline(
            SimpleImputer(strategy='constant', fill_value=0),
            # needed for linear models with regularization
            StandardScaler()
        )

        num_proc_nlin = make_pipeline(SimpleImputer(strategy='constant', fill_value=0))

        # categorical
        cat_proc_nlin = make_pipeline(
            SimpleImputer(missing_values=None, strategy='constant', fill_value=missing_cat),
            # do not use OneHOtEncoder for nonlinear estimator
            OrdinalEncoder(categories=categories)
        )

        cat_proc_lin = make_pipeline(
            SimpleImputer(missing_values=None, strategy='constant', fill_value=missing_cat),
            OneHotEncoder(categories=categories)
        )

        # transformation to use for non-linear estimators
        processor_nlin = make_column_transformer(
            (cat_proc_nlin, cat_cols),
            (num_proc_nlin, num_cols),
            remainder='passthrough')

        # transformation to use for linear estimators
        processor_lin = make_column_transformer(
            (cat_proc_lin, cat_cols),
            (num_proc_lin, num_cols),
            remainder='passthrough')

        return processor_lin, processor_nlin

    @staticmethod
    def build_classifier_estimators(processor_lin, processor_nlin, list_of_estimators=[]):
        linear_est = [
            ('Logi', LogisticRegression(max_iter=10000)),
            # Ridge doesn't have predict_proba
            # ('ridge', RidgeClassifier(class_weight='balanced')),
        ]

        nonlinear_est = [
            # ('LGBM_bal', LGBMClassifier(class_weight='balanced')),
            ('LGBM', LGBMClassifier()),
            ('RandomForest', RandomForestClassifier(random_state=42)),
            ## much faster than GradientBoostingClassifier for large dataset
            ('HistGradientBoosting', HistGradientBoostingClassifier(random_state=0)),
            ## An AdaBoost classifier is a meta-estimator, default base_estimator=DecisionTreeClassifier
            ('AdaBoosting', AdaBoostClassifier(learning_rate=0.5, n_estimators=100)),
            ## A Bagging classifier is an ensemble meta-estimator, default base_estimator=DecisionTreeClassifier
            ('Bagging', BaggingClassifier(n_estimators=10)),
            ## GaussianNB, KNN generally don't have good performance
            ##('GaussianNB', GaussianNB()),
            ##('KNN', KNeighborsClassifier(n_neighbors=3)),
        ]

        estimators = []
        for name, est in linear_est:
            if len(list_of_estimators) == 0 or name in list_of_estimators:
                estimators.append((name, make_pipeline(processor_lin, est)))

        for name, est in nonlinear_est:
            if len(list_of_estimators) == 0 or name in list_of_estimators:
                estimators.append((name, make_pipeline(processor_nlin, est)))
    
        ## stacking ensemble
        stacking_classifier = StackingClassifier(estimators=estimators,
                                                 final_estimator=LGBMClassifier(class_weight='balanced'),
                                                 stack_method='predict_proba')

        return estimators, stacking_classifier
