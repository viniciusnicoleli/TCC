# For all Class
import smote_variants as sv
import pandas as pd
import numpy as np
import sklearn as sk

# Getting utilidades
import os, sys
sys.path.insert(0, os.path.abspath(".."))
from utilidades.calibration import utilities as ult


# For function fit
import sklearn.pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import Pipeline

# LGBM
from skopt.space import Real, Categorical, Integer
from skopt import BayesSearchCV
from lightgbm import early_stopping
from lightgbm import LGBMClassifier
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import average_precision_score


class dbsmote_tcc():
    def __init__(self, dataframe : pd.DataFrame, target: str,metric : str = 'average_precision', pipe_final : sklearn.pipeline = None):
        self.dataframe = dataframe
        self.target = target
        self.metric = metric
        self._pipe_final = pipe_final
    
    def fit(self,random_state=42):

        X,y = ult.splitxy(self.dataframe,self.target)

        self.X_train, self.y_train, self.X_test, self.y_test, self.X_val, self.y_val = ult.train_test_val(X,y)

        prep_feat_tuple = ult.create_prep_pipe2(self.dataframe,self.target)
        self.prep_feat = prep_feat_tuple[0]

        self.lists_pandarizer = list(prep_feat_tuple[1]) + list(prep_feat_tuple[2])

        self.X_train_dbsmote, self.y_train_dbsmote = self.dbsmote()

        self.pipe_prep = Pipeline([
                    ('transformer_prep', self.prep_feat),
                    ("pandarizer", FunctionTransformer(lambda x: pd.DataFrame(x, columns = self.lists_pandarizer))),
                ])
        self.pipe_prep.fit(self.X_train_dbsmote)
        
        LGBM = LGBMClassifier(random_state = 42, n_jobs = -1)

        pipe_tuning = Pipeline([
            ('transformer_prep', self.prep_feat),
            ("pandarizer", FunctionTransformer(lambda x: pd.DataFrame(x, columns = self.lists_pandarizer))),
            ('estimator', LGBM)
        ])
        
        cv = StratifiedShuffleSplit(n_splits = 5, test_size = 0.3, random_state = 42)
        
        metric = self.metric
        
        fit_params = {
            'eval_metric': metric, 
            'eval_set': [(self.X_test, pd.DataFrame(self.y_test))],
            'callbacks': [(early_stopping(stopping_rounds = 10, verbose = True))],
        }        
        
        LGBM_search_space = {
            "estimator__learning_rate": Real(0.001, 0.01, prior = 'log-uniform'),
            "estimator__n_estimators": Integer(100, 1000),
            "estimator__class_weight": Categorical(['balanced', None]),
            "estimator__num_leaves": Integer(32, 256),
            "estimator__min_child_samples": Integer(100, 1000),
            "estimator__reg_alpha": Real(0, 100, prior = 'uniform'),
            "estimator__reg_lambda": Real(10., 200., prior = 'uniform'),
            "estimator__objective": Categorical(['binary']),
            "estimator__importance_type":Categorical(['gain']),
            "estimator__boosting_type": Categorical(['goss'])
        }    
        
        LGBM_bayes_search = BayesSearchCV(pipe_tuning, LGBM_search_space, n_iter = 2, scoring = metric, 
                                         return_train_score = True, 
                                         fit_params = fit_params,
                                         n_jobs = -1, cv = cv, random_state = random_state, optimizer_kwargs = {'base_estimator': 'GP'})
        
        
        LGBM_bayes_search.fit(self.pipe_prep.transform(self.X_train_dbsmote), self.y_train_dbsmote)        

        results_cv = pd.DataFrame(LGBM_bayes_search.cv_results_)
        
        temp = results_cv[['mean_train_score', 'mean_test_score']]
        temp['diff'] = temp['mean_test_score'] - temp['mean_train_score']
        to_go = temp[abs(temp['diff']) < 0.05].sort_values(by = 'mean_test_score', ascending = False).head(1).index
        
        params = results_cv.loc[to_go.values[0]]
        kwargs = params.params   
        print(kwargs)
        
        best_LGBM = LGBMClassifier(random_state = random_state, n_jobs = -1, verbose = -1, **kwargs)
        
        best_LGBM.fit(self.pipe_prep.transform(self.X_train_dbsmote), self.y_train_dbsmote, early_stopping_rounds = 10, verbose = 20, eval_metric = metric,
                     eval_set = [(self.pipe_prep.transform(self.X_test), self.y_test)]) 
        
        
        pipe_final = Pipeline(
        [
            ('pipe_transformer_prep', self.pipe_prep),
            ('pipe_estimator', best_LGBM)
        ])       
        
        self._pipe_final = pipe_final
        
    def dbsmote(self):
        
        Dbsmote = sv.DBSMOTE(random_state=42)
        X_train_dbsmote, y_train_dbsmote = Dbsmote.sample(self.X_train,
                                                             self.y_train)

        X_train_dbsmote = pd.DataFrame(X_train_dbsmote,columns=self.X_train.columns)
        return(X_train_dbsmote,y_train_dbsmote)

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
        
