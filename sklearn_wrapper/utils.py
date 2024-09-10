from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.linear_model import LogisticRegressionCV
from scipy.stats import ks_2samp
from sklearn.metrics import roc_curve, roc_auc_score, auc
import numpy as np
import pandas as pd

from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, LabelEncoder
from sklearn.preprocessing import StandardScaler

import time
from functools import reduce, wraps

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

def varclus_scipy(df, max_num_clusters=100):
    from scipy.stats import spearmanr
    from scipy.cluster import hierarchy

    corr = spearmanr(df).correlation
    corr_linkage = hierarchy.ward(corr)
#     print(corr)
#     print(corr_linkage.mean())

    # range of number of clusters
    # define maximum number of clusters
    n_min, n_max = df.shape[1]//6, df.shape[1]//5
    if n_min > max_num_clusters:
        n_min, n_max = max_num_clusters*0.9, max_num_clusters*1.1

    print("INFO: number of clusters range", n_min, n_max)
    # dynamicaly adjust <thr> to achieve desired number of clusters
    thr, thr_above, thr_below = 0.1, 1, 20
    while True:
        cluster_ids = hierarchy.fcluster(corr_linkage, thr, criterion='distance')
        if len(np.unique(cluster_ids)) > n_max:
            thr, thr_above = thr*2, thr
        elif len(np.unique(cluster_ids)) < n_min:
            thr, thr_below = (thr+thr_above)/2, thr
        else:
            break
        print('\t', thr, 'above', thr_above, 'below', thr_below, len(np.unique(cluster_ids)))

    df_cluster = pd.DataFrame({'variable': df.columns, 'cluster': cluster_ids}).sort_values('cluster')
    print('INFO: final number of clusters', len(np.unique(cluster_ids)), f'thr {thr:.3f}')
    return df_cluster

def gen_cap_curve(y_act, y_pred, wt, name, tag, tag2=None, verb=False):
    ''' weighted version of generating performance & capture curve
        return: pandas dataframe
    '''
    if wt is None:
        wt = np.ones(len(y_act))
    if verb:
        print(f'avgWt: {np.mean(wt):.4f}')
    fpr, tpr, _ = roc_curve(y_act, y_pred, sample_weight=wt)
    auc0 = auc(fpr, tpr)
    auc2 = roc_auc_score(y_act, y_pred, sample_weight=wt)
    # print(f'auc0:{auc0:.4f} auc2:{auc2:.4f}')
    ks = np.max(tpr-fpr)*100

    capt = pd.DataFrame({'actual': y_act, 'pred': y_pred, 'wt': wt})
    capt.sort_values(['pred'], ascending=False, inplace=True)
    capt.reset_index(drop=True, inplace=True)

    capt['actualCum'] = capt.eval('actual*wt').cumsum()/capt.eval('actual*wt').sum()
    capt['wtCum'] = capt['wt'].cumsum()/capt['wt'].sum()

    idx = [np.max(np.where(capt['wtCum'] <= w0)[0]) for w0 in [0.05, 0.10, 0.20]]

    if verb:
        print('wtCum:', capt.loc[idx, 'wtCum'].values)
        print('captCum:', capt.loc[idx, 'actualCum'].values)

    capt5, capt10, capt20 = capt.loc[idx, 'actualCum'].values

    count, wtCount = len(wt), np.sum(wt)

    x0 = np.arange(2.5, 102.5, 2.5)/100
    # x0 = x0[x0<=1]
    idx = [np.max(np.where(capt['wtCum'] <= w0)[0]) for w0 in x0]

    popCap = pd.DataFrame({'x': x0, 'y': capt.loc[idx, 'actualCum'], 'Model': name, 'Type': tag})
    rocCurve = pd.DataFrame({'fpr': fpr, 'tpr': tpr, 'Model': name, 'Type': tag})
    if tag2 is not None:
        popCap['SubType'] = tag2
        rocCurve['SubType'] = tag2

    ids = np.arange(0, len(fpr))
    jx = [int(v) for v in np.percentile(ids, q=np.arange(0, 105, 5))]
    rocCurve = rocCurve.loc[jx]

    def _calc_bin_cap():
        capt = pd.DataFrame({'actual': y_act, 'pred': y_pred, 'wt': wt})
        capt.sort_values(['pred'], ascending=False, inplace=True)
        capt.reset_index(drop=True, inplace=True)
        capt['actualCum'] = capt.eval('actual*wt').cumsum()
        capt['wtCum'] = capt['wt'].cumsum()
        capt['wtCum%'] = capt['wtCum']/capt['wt'].sum()

        x0 = np.arange(10, 110, 10)/100
        idx = [np.max(np.where(capt['wtCum%'] <= w0)[0]) for w0 in x0]

        # group into bins
        binCap = capt.iloc[idx].reset_index(drop=True)
        binCap['actualCum_p1'] = binCap['actualCum'].shift(1)
        binCap['wtCum_p1'] = binCap['wtCum'].shift(1)

        binCap['actualBin'] = binCap['actualCum'] - binCap['actualCum_p1'].fillna(0)
        binCap['wtBin'] = binCap['wtCum'] - binCap['wtCum_p1'].fillna(0)
        binCap['capBin%'] = binCap.eval('actualBin/wtBin*100')
        binCap['Model'] = name
        binCap['Type'] = tag
        if tag2 is not None:
            binCap['SubType'] = tag2

        binCap = (binCap.reset_index()
                  .rename(columns={'capBin%': 'bad_rate%', 'index': 'bin', 'actualBin': 'bads',
                                   'wtBin': 'total',  # weighted count
                                   'wtCum': 'cum_total',
                                   'actualCum': 'cum_bad'}))
        binCap['cum_good'] = binCap.eval('cum_total-cum_bad')
        binCap['cum_good_p1'] = binCap['cum_good'].shift(1)
        binCap['goods'] = binCap['cum_good']-binCap['cum_good_p1'].fillna(0)

        # return binCap[['actualBin','wtBin','capBin%','Model','Type','SubType']].reset_index(drop=True)
        return binCap[['bin', 'bads', 'goods', 'total', 'bad_rate%', 'cum_bad', 'cum_good', 'Model', 'Type', 'SubType']]

    # import pdb; pdb.set_trace()
    binCap = _calc_bin_cap()

    return auc0, ks, capt5, capt10, capt20, count, wtCount, popCap, rocCurve, binCap

def gen_cap_curve_equal_weigth(y_act, y_pred, tag='model', name='type'):
    ''' return: pandas dataframe
    '''
    fpr, tpr, _ = roc_curve(y_act, y_pred)
    auc0 = auc(fpr, tpr)
    # ks = ks_2samp(y_act, y_pred)
    ks0 = np.max(tpr-fpr)*100

    capt = pd.DataFrame({'actual': y_act, 'pred': y_pred})
    capt.sort_values(['pred'], ascending=False, inplace=True)
    capt['actualCum'] = capt['actual'].cumsum()/capt['actual'].sum()
    capt['actualCum'].quantile(q=[0.05, 0.10, 0.20])
    capt5, capt10, capt20 = capt['actualCum'].quantile(q=[0.05, 0.10, 0.20]).values

    x0 = np.arange(2.5, 102.5, 2.5)/100
    x0 = x0[x0 <= 1]
    popCap = pd.DataFrame({'x': x0, 'y': capt['actualCum'].quantile(q=x0), 'Model': name, 'Type': tag})
    rocCurve = pd.DataFrame({'fpr': np.percentile(fpr, q=np.arange(0, 105, 5)),
                             'tpr': np.percentile(tpr, q=np.arange(0, 105, 5)),
                             'Model': name, 'Type': tag})
    return auc0, ks0, capt5, capt10, capt20, popCap, rocCurve


''' ############################################################################################################ '''
''' ############################################################################################################ '''

@timethis
def run_classifier_test(N=1000, p=10, p2=3, perform_cv=True, **kwargs):
    ''' generate random classfication dataset to test stacking classifier
        Additional kwargs:
           include_stacking=False,
    '''
    from sklearn.datasets import make_classification
    pd.set_option('display.width', 150)
    pd.set_option('display.max_columns', 20)

    XX, yy = make_classification(N, p, n_informative=p2)
    XX, yy = pd.DataFrame(XX), pd.Series(yy)
    XX.columns = [f'V{v:02d}' for v in range(p)]

    wt = np.ones(len(yy))
    # more
    wt[yy == 0] = 5

    clf, stats0, popCaps0, binCaps0 = run_classifier_test_XY(XX, yy, wt, perform_cv, **kwargs)

    # print('\n\n\n')
    # print('************************ Variable Clustering Algorithm ************************')
    # print('\n\n\n')

    # df_clus = varclus_scipy(XX, 10)
    # df_clus['rank'] = df_clus.groupby('cluster').cumcount()+1
    # vars_selected = df_clus.query('rank<=2')['variable']
    # print(f'INFO: {len(vars_selected)} variables out of {XX.shape[1]} selected')

    # import copy
    # XX_copy = copy.copy(XX.iloc[:, vars_selected])
    # ''' reset columns to eliminate ValueError: max_features must be in '''
    # XX_copy.columns = [f'v{i}' for i in np.arange(XX_copy.shape[1])]

    # run_classifier_test_XY(XX_copy, yy, wt, perform_cv, **kwargs)

    return XX, yy, clf, stats0, popCaps0, binCaps0

def run_classifier_test_XY(XX, yy, wt,
                           perform_cv=True,
                           ** kwargs):
    ''' generate random classfication dataset to test stacking classifier
        Additional kwargs:
           include_stacking=False,
    '''
    from sklearn_wrapper.estimators import classifier as sk_classifier
    from sklearn.model_selection import train_test_split
    pd.set_option('display.width', 150)
    pd.set_option('display.max_columns', 20)
    pd.set_option('display.precision', 2)

    # import pdb; pdb.set_trace()

    print(f'INFO: bad rate {np.sum(wt*yy)/np.sum(wt):.4f}, wt {np.mean(wt)}')

    X_train, X_test, y_train, y_test, wt_train, wt_test = train_test_split(XX, yy, wt, test_size=0.3, random_state=943)

    temp = [XX, X_train, y_train, wt_train, X_test, y_test, wt_test]

    print(f'INFO: pre run wt_train {np.mean(wt_train):.3f} wt_test {np.mean(wt_test):.3f}')

    clf = sk_classifier(temp, excl_model_cols=[], bench_scores=[])

    clf.run(perform_cv=perform_cv, **kwargs)

    print(f'INFO: post run wt_train {np.mean(wt_train):.3f} wt_test {np.mean(wt_test):.3f}')

    stats0, popCaps0, binCaps0 = clf.gen_perf_stats(**kwargs)

    print('Summary statistics')
    print(stats0)

    print('Performance by bin')
    print(binCaps0.pivot_table(index=['Type', 'bin'], columns=['Model'], values=['bad_rate%']))

    return clf, stats0, popCaps0, binCaps0

def plot_performance_curves(popCaps0, binCaps0)
    # if 'plot' in kwargs.keys() and kwargs['plot'] == True:
    from plotnine import qplot, geom_line, geom_point, facet_wrap, aes, labs
    obj = qplot('x', 'y', data=popCaps0) + geom_line() + aes(color='factor(Model)') + facet_wrap('Type') + labs(title='Population Capture')
    # binCaps0['bin'] = binCaps0['bin'].astype('str')

    obj2 = qplot('bin', 'bad_rate%', data=binCaps0[['bin', 'bad_rate%', 'Model', 'Type']]) + \
        geom_line() + aes(color='factor(Model)') + facet_wrap('Type') + labs(title='Bin Bad Rate')
    
    print(obj)
    print(obj2)

    from collections import namedtuple

    PerfCurves = namedtuple('PerfCurves', 'popCapCurves, badCapCurves')
    return PerfCurves(popCapCurves=obj, badCapCurves=obj2)