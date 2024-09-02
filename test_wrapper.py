from sklearn_wrapper.utils import *

# generate <N> random records,
# <p> features, <p2> features with information
# perform_cv=True: perform cross-validation
# include_stacking=True: building ensemble stacking classifier
run_classifier_test(N=10000, p=20, p2=5, perform_cv=True, include_stacking=True)
