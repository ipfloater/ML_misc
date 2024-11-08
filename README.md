# Machine Learning Classifier Wrapper

## Note:
* Code refactored into `sklear_ML` folder
  * `from sklearn_ML.ensemble_classifier import EnsembleClassifier`
  * `from sklearn_ML.pipeline_utils import PipelineBuilder`
* see [`test_model_performance.ipynb`](./test_model_performance.ipynb) for unit tests:
  * pipeline test
  * ensemble classifier test
  * `roc_auc` curve test
  * `precision_recall` curve test
* Original code still in `sklearn_wrapper` folder:
  *  see [`test_wrapper.py`](./test_wrapper.py) for original test notebook

## Input Data:
* `XX`: `pandas` dataframe
* `yy`: `pandas` Series with binary values `0` and `1`

## Data Pre-processing & Basic Feature Engineerings
* Columns types are auto-detected
* **Missing Value Imputations & Transformations**
  * Using `sklearn.impute.SimpleImputer` 
  * For numeric columns:
    *  Missing values imputed to `0` 
    *  Normalized to accomodate linear model with regularization
  * For categorical columns:
    * Missing values are imputed as a special category/level = `missing`
    * Use `OrdinalEncoder` to transform into numerical values
* **Pipeline processing** 
  * Apply the following Scikit-Learn pipelines:
    * `sklearn.pipeline.make_pipeline`
    * `sklearn.compose.make_column_transformer`

## Advanced Feature Engineering (Not Supported!)
* Because this requires domain knowledge and is data source dependent

## Model Features
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

## Performance Summary Features
* Option to generate Cross-Validation performance
* Generate Test Performance
  * Report similar to Traditional Risk Scorecard
  * Bad rate by (10) bins
  * Cumulative Bad rate by (10) bins

## Test
* The library has been tested under Ubuntu 22.04, as well as in Google Colab
* run `test_wrapper.ipynb` in Google Colab (recommended!)
* run `test_wrapper.py` in linux terminal
