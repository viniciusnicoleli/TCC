
# Warnings
import warnings
warnings.filterwarnings('ignore')

# Importing utilities
import os, sys
sys.path.insert(0, os.path.abspath(".."))

# Basic imports
import numpy as np
import pandas as pd
import collections
import sklearn

# Model
from utilidades.calibration import utilities as ult
from sklearn.tree import DecisionTreeClassifier
from skopt import BayesSearchCV
from skopt.space import Real, Categorical, Integer
from lightgbm import early_stopping
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.model_selection import StratifiedShuffleSplit





class tcc_dt():
    def __init__(self, dataframe : pd.DataFrame, target: str, X_test : pd.DataFrame, y_test = pd.Series, metric : str = 'average_precision', pipe_final : sklearn.pipeline = None):
        self.dataframe = dataframe
        self.target = target
        self.X_test = X_test
        self.y_test = y_test
        self.metric = metric
        self._pipe_final = pipe_final

    def bootstrap_rows(self,df : pd.DataFrame, column : str, n : int = 4):
        tamanho = int(np.round(df[df[column] == 1].shape[0]/n))
        df_to_go = df[df[column] == 1].iloc[np.random.randint(tamanho, size=tamanho)]
        return(df_to_go)
    
    def fit(self):
        
        bootstrap_base = self.bootstrap_rows(self.dataframe, 'to_bootstrap')
        self.dataframe = self.dataframe.drop(['to_bootstrap'], axis=1)
        bootstrap_base = bootstrap_base.drop(['to_bootstrap'], axis=1)
        
        self.dataframe = pd.concat([self.dataframe, bootstrap_base], axis = 0)
        
        X_train, y_train = ult.splitxy(self.dataframe, self.target)

        prep_feat_tuple = ult.create_prep_pipe(self.dataframe, self.target)
        prep_feat = prep_feat_tuple[0]
    
        lists_pandarizer = list(prep_feat_tuple[1]) + list(prep_feat_tuple[2])
        
        DT = DecisionTreeClassifier(random_state = 42)

        pipe_tuning = Pipeline([
            ('transformer_prep', prep_feat),
            ("pandarizer", FunctionTransformer(lambda x: pd.DataFrame(x, columns = lists_pandarizer))),
            ('estimator', DT)
        ])
        
        pipe_prep = Pipeline([
            ('transformer_prep', prep_feat),
            ("pandarizer", FunctionTransformer(lambda x: pd.DataFrame(x, columns = lists_pandarizer))),
        ])
        pipe_prep.fit(X_train)       
        
        cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.3, random_state = 42)
        
        metric = self.metric
        
        fit_params = {
            'eval_metric': metric, 
            'eval_set': [(self.X_test, pd.DataFrame(self.y_test))],
            'callbacks': [(early_stopping(stopping_rounds = 10, verbose = True))],
        }  
        
        #LGBM_search_space = {
        #    "estimator__learning_rate": Real(0.001, 0.01, prior = 'log-uniform'),
        #    "estimator__n_estimators": Integer(100, 1000),
        #    "estimator__class_weight": Categorical(['balanced', None]),
        #    "estimator__num_leaves": Integer(32, 256),
        #    "estimator__min_child_samples": Integer(100, 1000),
        #    "estimator__reg_alpha": Real(0, 100, prior = 'uniform'),
        #    "estimator__reg_lambda": Real(10., 200., prior = 'uniform'),
        #    "estimator__objective": Categorical(['binary']),
        #    "estimator__importance_type":Categorical(['gain']),
        #    "estimator__boosting_type": Categorical(['goss'])
        #}    
        #
        #LGBM_bayes_search = BayesSearchCV(pipe_tuning, LGBM_search_space, n_iter = 2, scoring = metric, 
        #                                 return_train_score = True, 
        #                                 fit_params = fit_params,
        #                                 n_jobs = -1, cv = cv, random_state = 42, optimizer_kwargs = {'base_estimator': 'GP'})
        #
        #
        #LGBM_bayes_search.fit(X_train, y_train)        
        #
        #results_cv = pd.DataFrame(LGBM_bayes_search.cv_results_)     
        
        DT_search_space = {
            "estimator__criterion": Categorical(['gini', 'entropy', 'log_loss']),
            "estimator__max_depth": Integer(5, 100),
            "estimator__min_samples_split": Integer(10, 1000),
            "estimator__min_samples_leaf": Integer(10, 1000),
            "estimator__ccp_alpha": Real(0.0001, 1, prior = 'log-uniform')
        }    
        
        DT_bayes_search = BayesSearchCV(pipe_tuning, DT_search_space, n_iter = 32, scoring = metric, 
                                         return_train_score = True, 
                                         fit_params = fit_params,
                                         n_jobs = -1, cv = cv, random_state = 42, optimizer_kwargs = {'base_estimator': 'GP'})
        
        
        DT_bayes_search.fit(X_train, y_train)        
        
        results_cv = pd.DataFrame(DT_bayes_search.cv_results_)
        
        temp = results_cv[['mean_train_score', 'mean_test_score']]
        temp['diff'] = temp['mean_test_score'] - temp['mean_train_score']
        to_go = temp[abs(temp['diff']) < 0.20].sort_values(by = 'mean_test_score', ascending = False).head(1).index
        
        
        params = results_cv.loc[to_go.values[0]]
        kwargs = params.params   
        kwargs = collections.OrderedDict((key.replace('estimator__', ''), value) for key, value in kwargs.items())
        print(kwargs)
        
        
        best_DT = DecisionTreeClassifier(random_state = 42, **kwargs)
        
        best_DT.fit(pipe_prep.transform(X_train), y_train) 

        
        pipe_final = Pipeline(
        [
            ('pipe_transformer_prep', pipe_prep),
            ('pipe_estimator', best_DT)
        ])       
        
        self._pipe_final = pipe_final
        return(self._pipe_final)
