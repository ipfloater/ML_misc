from sklearn_wrapper.utils import *

# generate <N> random records,
# <p> features, <p2> features with information
# perform_cv=True: perform cross-validation
# include_stacking=True: building ensemble stacking classifier

XX, yy, clf, stats0, popCaps0, binCaps0 = run_classifier_test(N=5000, p=20, p2=5, perform_cv=True, include_stacking=True, plot=True)

# optional: generate & display performance curves
# res = plot_performance_curves(popCaps0, binCaps0)
# print(res.popCapCurves)
# print(res.badCapCurves)

