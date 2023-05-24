# Warnings
import warnings
warnings.filterwarnings('ignore')

# Importing utilities
import os, sys
sys.path.insert(0, os.path.abspath(".."))

# Folder imports
from src.BBO import dt_for_tcc as dff
from src.BBO import indices_tcc as itc 

# Basic imports
import numpy as np
import pandas as pd
import sklearn

# Model
from utilidades.calibration import utilities as ult
from sklearn.metrics import average_precision_score



class tcc_proposto():
    def __init__(self, dataframe : pd.DataFrame, target: str, metric : str = 'average_precision', pipe_final : sklearn.pipeline = None):
        self.dataframe = dataframe
        self.target = target
        self.metric = metric
        self._pipe_final = pipe_final
    
    def fit(self):
        X, y = ult.splitxy(self.dataframe, self.target)
        X_train, y_train, X_test, y_test, X_val, y_val = ult.train_test_val(X,y)
        df_train = pd.merge(X_train, pd.DataFrame(y_train),left_index=True, right_index=True, how = 'inner')
        
        indices = itc.tcc_indices_to_resample(dataframe = df_train, target = self.target)
        indices.fit()
        indx = indices.indices
        
        df_train['to_bootstrap'] = 0
        df_train.iloc[indx]['to_bootstrap'] = 1
        
        pipe_final =  [
            dff.tcc_dt(df_train,X_test= X_test,y_test = y_test,target = 'Class').fit(),
            dff.tcc_dt(df_train,X_test= X_test,y_test = y_test,target = 'Class').fit(),
            dff.tcc_dt(df_train,X_test= X_test,y_test = y_test,target = 'Class').fit(),
            dff.tcc_dt(df_train,X_test= X_test,y_test = y_test,target = 'Class').fit(),
            dff.tcc_dt(df_train,X_test= X_test,y_test = y_test,target = 'Class').fit(),
            dff.tcc_dt(df_train,X_test= X_test,y_test = y_test,target = 'Class').fit(),
            dff.tcc_dt(df_train,X_test= X_test,y_test = y_test,target = 'Class').fit(),
            dff.tcc_dt(df_train,X_test= X_test,y_test = y_test,target = 'Class').fit(),
            dff.tcc_dt(df_train,X_test= X_test,y_test = y_test,target = 'Class').fit(),
            dff.tcc_dt(df_train,X_test= X_test,y_test = y_test,target = 'Class').fit()
        ]           
        
        self._pipe_final = pipe_final
        
    def predict_proba(self, who : str = 'val'):
        if self._pipe_final is None:
            return None
        
        X, y = ult.splitxy(self.dataframe, self.target)
        X_train, y_train, X_test, y_test, X_val, y_val = ult.train_test_val(X,y)        
        
        dic = {'val': X_val, 
              'test': X_test,
              'train': X_train}
        
        
        temp_0 = self._pipe_final[0].predict_proba(dic[who])[:,1]
        temp_1 = self._pipe_final[1].predict_proba(dic[who])[:,1]
        temp_2 = self._pipe_final[2].predict_proba(dic[who])[:,1]
        temp_3 = self._pipe_final[3].predict_proba(dic[who])[:,1]
        temp_4 = self._pipe_final[4].predict_proba(dic[who])[:,1]
        temp_5 = self._pipe_final[5].predict_proba(dic[who])[:,1]
        temp_6 = self._pipe_final[6].predict_proba(dic[who])[:,1]
        temp_7 = self._pipe_final[7].predict_proba(dic[who])[:,1]
        temp_8 = self._pipe_final[8].predict_proba(dic[who])[:,1]
        
        
        temp_0 = pd.DataFrame(temp_0, columns = ['predict'])
        temp_1 = pd.DataFrame(temp_1, columns = ['predict'])
        temp_2 = pd.DataFrame(temp_2, columns = ['predict'])
        temp_3 = pd.DataFrame(temp_3, columns = ['predict'])
        temp_4 = pd.DataFrame(temp_4, columns = ['predict'])
        temp_5 = pd.DataFrame(temp_5, columns = ['predict'])
        temp_6 = pd.DataFrame(temp_6, columns = ['predict'])
        temp_7 = pd.DataFrame(temp_7, columns = ['predict'])
        temp_8 = pd.DataFrame(temp_8, columns = ['predict'])
        
        temp = pd.concat([temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8], 1)
           
        y_score = temp.mean(axis=1).to_numpy()
        
        
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
        
        
        temp_0 = self._pipe_final[0].predict_proba(dic_x[who])[:,1]
        temp_1 = self._pipe_final[1].predict_proba(dic_x[who])[:,1]
        temp_2 = self._pipe_final[2].predict_proba(dic_x[who])[:,1]
        temp_3 = self._pipe_final[3].predict_proba(dic_x[who])[:,1]
        temp_4 = self._pipe_final[4].predict_proba(dic_x[who])[:,1]
        temp_5 = self._pipe_final[5].predict_proba(dic_x[who])[:,1]
        temp_6 = self._pipe_final[6].predict_proba(dic_x[who])[:,1]
        temp_7 = self._pipe_final[7].predict_proba(dic_x[who])[:,1]
        temp_8 = self._pipe_final[8].predict_proba(dic_x[who])[:,1]
        
        
        temp_0 = pd.DataFrame(temp_0, columns = ['predict'])
        temp_1 = pd.DataFrame(temp_1, columns = ['predict'])
        temp_2 = pd.DataFrame(temp_2, columns = ['predict'])
        temp_3 = pd.DataFrame(temp_3, columns = ['predict'])
        temp_4 = pd.DataFrame(temp_4, columns = ['predict'])
        temp_5 = pd.DataFrame(temp_5, columns = ['predict'])
        temp_6 = pd.DataFrame(temp_6, columns = ['predict'])
        temp_7 = pd.DataFrame(temp_7, columns = ['predict'])
        temp_8 = pd.DataFrame(temp_8, columns = ['predict'])
        
        temp = pd.concat([temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8], 1)
           
        y_score = temp.mean(axis=1).to_numpy()
        average_precision = average_precision_score(dic_y[who], y_score)
        return average_precision
    
    
    def plot_dist(self):
        if self._pipe_final is None:
            return None        

        X, y = ult.splitxy(self.dataframe, self.target)
        X_train, y_train, X_test, y_test, X_val, y_val = ult.train_test_val(X,y)        
        
        
        temp_0 = self._pipe_final[0].predict_proba(X_train)[:,1]
        temp_1 = self._pipe_final[1].predict_proba(X_train)[:,1]
        temp_2 = self._pipe_final[2].predict_proba(X_train)[:,1]
        temp_3 = self._pipe_final[3].predict_proba(X_train)[:,1]
        temp_4 = self._pipe_final[4].predict_proba(X_train)[:,1]
        temp_5 = self._pipe_final[5].predict_proba(X_train)[:,1]
        temp_6 = self._pipe_final[6].predict_proba(X_train)[:,1]
        temp_7 = self._pipe_final[7].predict_proba(X_train)[:,1]
        temp_8 = self._pipe_final[8].predict_proba(X_train)[:,1]
        
        
        temp_0 = pd.DataFrame(temp_0, columns = ['predict'])
        temp_1 = pd.DataFrame(temp_1, columns = ['predict'])
        temp_2 = pd.DataFrame(temp_2, columns = ['predict'])
        temp_3 = pd.DataFrame(temp_3, columns = ['predict'])
        temp_4 = pd.DataFrame(temp_4, columns = ['predict'])
        temp_5 = pd.DataFrame(temp_5, columns = ['predict'])
        temp_6 = pd.DataFrame(temp_6, columns = ['predict'])
        temp_7 = pd.DataFrame(temp_7, columns = ['predict'])
        temp_8 = pd.DataFrame(temp_8, columns = ['predict'])
        
        temp = pd.concat([temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8], 1)
           
        y_score_train = temp.mean(axis=1).to_numpy()

        
        temp_0 = self._pipe_final[0].predict_proba(X_val)[:,1]
        temp_1 = self._pipe_final[1].predict_proba(X_val)[:,1]
        temp_2 = self._pipe_final[2].predict_proba(X_val)[:,1]
        temp_3 = self._pipe_final[3].predict_proba(X_val)[:,1]
        temp_4 = self._pipe_final[4].predict_proba(X_val)[:,1]
        temp_5 = self._pipe_final[5].predict_proba(X_val)[:,1]
        temp_6 = self._pipe_final[6].predict_proba(X_val)[:,1]
        temp_7 = self._pipe_final[7].predict_proba(X_val)[:,1]
        temp_8 = self._pipe_final[8].predict_proba(X_val)[:,1]
        
        
        temp_0 = pd.DataFrame(temp_0, columns = ['predict'])
        temp_1 = pd.DataFrame(temp_1, columns = ['predict'])
        temp_2 = pd.DataFrame(temp_2, columns = ['predict'])
        temp_3 = pd.DataFrame(temp_3, columns = ['predict'])
        temp_4 = pd.DataFrame(temp_4, columns = ['predict'])
        temp_5 = pd.DataFrame(temp_5, columns = ['predict'])
        temp_6 = pd.DataFrame(temp_6, columns = ['predict'])
        temp_7 = pd.DataFrame(temp_7, columns = ['predict'])
        temp_8 = pd.DataFrame(temp_8, columns = ['predict'])
        
        temp = pd.concat([temp_0, temp_1, temp_2, temp_3, temp_4, temp_5, temp_6, temp_7, temp_8], 1)    
         
        y_score_val = temp.mean(axis=1).to_numpy()
        
        ult.plot_dist(y_train, y_score_train, y_val, y_score_val)
        