# Machine Learning Classifier Wrapper
## Features
* Based on Scikit-Learn
* Binary Classification Only
* Support the following estimators (aka ML algorithms):
  * Logistics Regression
  * LightGBM
  * RandomForest
  * HistGradientBoosting
  * AdaBoosting
  * Bagging
* Use `list_of_estimators` to pick & choose a subset of estimators from above:
  * for example `list_of_estimators=['Logi','LGBM','RandomForest']`
* Support Ensemble/Stacking classifier:
  * Optional, if `include_stacking=True`
* Option to generate Cross-Validation performance
* Generate Test Performance
  * Report similar to Traditional Risk Scorecard
  * Bad rate by (10) bins
  * Cumulative Bad rate by (10) bins

## Test
* The library has been tested under Ubuntu 22.04, as well as in Google Colab
* run `test_wrapper.ipynb` in Google Colab (recommended!)
* run `test_wrapper.py` in linux terminal
