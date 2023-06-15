# Warnings
import warnings
warnings.filterwarnings('ignore')

# Basic import
import sklearn
import numpy as np
import collections
import pandas as pd

# Importing utilities
import os, sys
sys.path.insert(0, os.path.abspath(".."))
from utilidades.calibration import utilities as ult

# For model.
from skopt import BayesSearchCV
from sklearn.ensemble import AdaBoostClassifier
from skopt.space import Real, Categorical, Integer
from lightgbm import early_stopping
from sklearn.pipeline import Pipeline
from imblearn.ensemble import EasyEnsembleClassifier
from sklearn.preprocessing import FunctionTransformer
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedShuffleSplit

class tcc_easyemsemble():
    def __init__(self, dataframe : pd.DataFrame, target: str, metric : str = 'average_precision', pipe_final : sklearn.pipeline = None):
        self.dataframe = dataframe
        self.target = target
        self.metric = metric
        self._pipe_final = pipe_final
    
    def fit(self):
        X, y = ult.splitxy(self.dataframe, self.target)

        X_train, y_train, X_test, y_test, X_val, y_val = ult.train_test_val(X,y)
    
        prep_feat_tuple = ult.create_prep_pipe(self.dataframe, self.target)
        prep_feat = prep_feat_tuple[0]
    
        lists_pandarizer = list(prep_feat_tuple[1]) + list(prep_feat_tuple[2])
        
        pipe_prep = Pipeline([
            ('transformer_prep', prep_feat),
            ("pandarizer", FunctionTransformer(lambda x: pd.DataFrame(x, columns = lists_pandarizer))),
        ])
        pipe_prep.fit(X_train)       
        
        cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.3, random_state = 42)
        
        metric = self.metric
        
        ## AdaBoost

        ADA = AdaBoostClassifier(random_state = 42)

        pipe_tuning = Pipeline([
            ('transformer_prep', prep_feat),
            ("pandarizer", FunctionTransformer(lambda x: pd.DataFrame(x, columns = lists_pandarizer))),
            ('estimator', ADA)
        ])        
        
        fit_params = {
            'eval_metric': metric, 
            'eval_set': [(X_test, pd.DataFrame(y_test))],
            'callbacks': [(early_stopping(stopping_rounds = 10, verbose = True))],
        }        
        
        ADA_search_space = {
            "estimator__n_estimators": Integer(100, 1000),   
            "estimator__learning_rate": Real(0.001, 0.01, prior = 'log-uniform'),
            "estimator__algorithm": Categorical(['SAMME', 'SAMME.R'])
      }    
        
        ADA_bayes_search = BayesSearchCV(pipe_tuning, ADA_search_space, n_iter = 1, scoring = metric, 
                                         return_train_score = True, 
                                         fit_params = fit_params,
                                         n_jobs = -1, cv = cv, random_state = 42, optimizer_kwargs = {'base_estimator': 'GP'})
        
        
        ADA_bayes_search.fit(X_train, y_train)        
        
        results_cv = pd.DataFrame(ADA_bayes_search.cv_results_)
        
        temp = results_cv[['mean_train_score', 'mean_test_score']]
        temp['diff'] = temp['mean_test_score'] - temp['mean_train_score']
        to_go = temp[abs(temp['diff']) < 0.2].sort_values(by = 'mean_test_score', ascending = False).head(1).index
        
        params = results_cv.loc[to_go.values[0]]
        kwargs = params.params   
        kwargs = collections.OrderedDict((key.replace('estimator__', ''), value) for key, value in kwargs.items())
        print(kwargs)
        
        best_ADA = AdaBoostClassifier(random_state = 42, **kwargs)
        
        ## EasyEmsemble 

        
        EASY = EasyEnsembleClassifier(random_state = 42, base_estimator = best_ADA)

        pipe_tuning = Pipeline([
            ('transformer_prep', prep_feat),
            ("pandarizer", FunctionTransformer(lambda x: pd.DataFrame(x, columns = lists_pandarizer))),
            ('estimator', EASY)
        ])
        
        
        fit_params = {
            'eval_metric': metric, 
            'eval_set': [(X_test, pd.DataFrame(y_test))],
            'callbacks': [(early_stopping(stopping_rounds = 10, verbose = True))],
        }        
        
        EASY_search_space = {
            "estimator__n_estimators": Integer(100, 1000),
            "estimator__warm_start": Categorical([True, False]),
            "estimator__sampling_strategy": Categorical(['majority', 'all']),
            "estimator__replacement": Categorical([True, False])
        }    
        
        print('chegamos aqui')
        EASY_bayes_search = BayesSearchCV(pipe_tuning, EASY_search_space, n_iter = 1, scoring = metric, 
                                         return_train_score = True, 
                                         fit_params = fit_params,
                                         n_jobs = -1, cv = cv, random_state = 42, optimizer_kwargs = {'base_estimator': 'GP'})
        
        
        EASY_bayes_search.fit(X_train, y_train)        
        
        results_cv = pd.DataFrame(EASY_bayes_search.cv_results_)
        
        temp = results_cv[['mean_train_score', 'mean_test_score']]
        temp['diff'] = temp['mean_test_score'] - temp['mean_train_score']
        to_go = temp.sort_values(by = 'mean_test_score', ascending = False).head(1).index
        
        #[abs(temp['diff']) < 0.6] 

        params = results_cv.loc[to_go.values[0]]
        kwargs = params.params   
        kwargs = collections.OrderedDict((key.replace('estimator__', ''), value) for key, value in kwargs.items())
        print(kwargs)
        
        best_EASY = EasyEnsembleClassifier(random_state = 42,  base_estimator = best_ADA,  **kwargs)
        
        best_EASY.fit(pipe_prep.transform(X_train), y_train) 
        
        
        pipe_final = Pipeline(
        [
            ('pipe_transformer_prep', pipe_prep),
            ('pipe_estimator', best_EASY)
        ])       
        
        self._pipe_final = pipe_final
        
    def predict_proba(self, who : str = 'val'):
        if self._pipe_final is None:
            return None
        
        X, y = ult.splitxy(self.dataframe, self.target)
        X_train, y_train, X_test, y_test, X_val, y_val = ult.train_test_val(X,y)        
        
        dic = {'val': X_val, 
              'test': X_test,
              'train': X_train}
        y_score = self._pipe_final.predict_proba(dic[who])[:,1]
        return y_score
    
    def get_metric(self, who : str = 'val'):
        if self._pipe_final is None:
            return None
        
        X, y = ult.splitxy(self.dataframe, self.target)
        X_train, y_train, X_test, y_test, X_val, y_val = ult.train_test_val(X,y)
        
        dic_x = {'val': X_val, 
              'test': X_test,
              'train': X_train}

        dic_y = {'val': y_val, 
              'test': y_test,
              'train': y_train}
        
        
        y_score = self._pipe_final.predict_proba(dic_x[who])[:,1]
        average_precision = average_precision_score(dic_y[who], y_score)
        return average_precision
    
    
    def plot_dist(self):
        if self._pipe_final is None:
            return None        

        X, y = ult.splitxy(self.dataframe, self.target)
        X_train, y_train, X_test, y_test, X_val, y_val = ult.train_test_val(X,y)        
        
        y_score_train = self._pipe_final.predict_proba(X_train)[:,1]
        y_score_val = self._pipe_final.predict_proba(X_val)[:,1]
        
        ult.plot_dist(y_train, y_score_train, y_val, y_score_val)
        