''' Utility class based on sklearn Pipeline
'''

from sklearn.pipeline import make_pipeline
from sklearn.compose import make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder

class PipelineBuilder:
    ''' pipeline builder class
    '''
    @staticmethod
    def build_pipeline(xx_all, excl_model_cols=None):
        ''' xx_all should contain all possible values for categoricals,
            otherwise pipeline will complain in processing X_test
        '''
        if excl_model_cols is None:
            excl_model_cols = []

        def extract_col_types(df_x, missing_cat):
            ''' missing_cat is the imputing string for categorical NA
            '''
            cat_cols = df_x.columns[df_x.dtypes == 'O']
            num_cols = df_x.columns[df_x.dtypes != 'O']

            # OneHotEncoder applied to linear estimator only!
            categories = [[v if v is not None else missing_cat for v in df_x[column].unique()] for column in df_x[cat_cols]]
            return num_cols, cat_cols, categories

        missing_cat = 'missing'

        # make sure to use entire sample to build categories for categorical variables!
        num_cols, cat_cols, categories = extract_col_types(xx_all.drop(columns=excl_model_cols), missing_cat)

        # numeric
        num_proc_lin = make_pipeline(
            SimpleImputer(strategy='constant', fill_value=0),
            StandardScaler()
        )

        num_proc_nlin = make_pipeline(SimpleImputer(strategy='constant', fill_value=0))

        # categorical
        cat_proc_nlin = make_pipeline(
            SimpleImputer(missing_values=None, strategy='constant', fill_value=missing_cat),
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
