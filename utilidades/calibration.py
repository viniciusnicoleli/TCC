import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import FeatureUnion
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime
import csv
import os

class utilities():    
    def splitxy(dataframe : pd.DataFrame, y : str = 'target'):
        X = dataframe.drop(labels = [y], axis = 1)
        y = dataframe[y]
        
        return(X, y)


    def train_test_val(X: pd.DataFrame, y: pd.Series):
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.33, random_state=42, stratify = y)
        X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.33, random_state=42, stratify = y_train)
        
        return(X_train, y_train, X_test, y_test, X_val, y_val)


    def create_prep_pipe(dataframe : pd.DataFrame, target_column : str):
        dataframe = dataframe.drop(labels = [target_column], axis = 1)
        num_cols = dataframe.select_dtypes(include=[float, int]).columns
        cat_cols = dataframe.select_dtypes(include=[object, np.datetime64]).columns
        
        if num_cols.values.shape[0] > 0:
            pipe_num = Pipeline(
            steps = [
                ("selector_num", ColumnTransformer([("selector", "passthrough", num_cols.values)], remainder = 'drop')),
                ('num_imputer', SimpleImputer(strategy='mean'))
            ])
        else:
            pipe_num = None
            
        if cat_cols.values.shape[0] > 0:
            pipe_cat = Pipeline(
            steps = [
                ("selector_cat", ColumnTransformer([("selector", "passthrough", cat_cols.values)], remainder = 'drop')),
                ('cat_imputer', SimpleImputer(strategy='most_frequent')),
                ("OrdinalEnc", OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = -1))
            ])        
        else:
            pipe_cat = None
        
        if pipe_num is not None and pipe_cat is not None:
            prep_feat = FeatureUnion(
            transformer_list = [
                ('num_pipe', pipe_num),
                ('cat_pipe', pipe_cat)
                
            ], verbose = False)  
            print(dataframe[cat_cols])
            return (prep_feat, num_cols.values, cat_cols.values)

        if pipe_num is not None and pipe_cat is None:
            prep_feat = FeatureUnion(
            transformer_list = [
                ('num_pipe', pipe_num)
                
            ], verbose = False)
            # print('jaggers',dataframe[cat_cols])
            return (prep_feat, num_cols.values, [])

        if pipe_num is None and pipe_cat is not None:
            prep_feat = FeatureUnion(
            transformer_list = [
                ('cat_pipe', pipe_cat)
                
            ], verbose = False) 
            return (prep_feat, [], cat_cols.values)

    def create_prep_pipe2(dataframe : pd.DataFrame, target_column : str):
        dataframe = dataframe.drop(labels = [target_column], axis = 1)
        num_cols = dataframe.select_dtypes(include=[float, int]).columns
        cat_cols = dataframe.select_dtypes(include=[object, np.datetime64]).columns

        print(num_cols,cat_cols)
        
        if num_cols.values.shape[0] > 0:
            pipe_num = Pipeline(
            steps = [
                ("selector_num", ColumnTransformer([("selector", "passthrough", num_cols.values)], remainder = 'drop')),
                ('num_imputer', SimpleImputer(strategy='mean')),
                ('standard_scaller', StandardScaler())
            ])
        else:
            pipe_num = None

        if cat_cols.values.shape[0] > 0:
            print('entrando no cat')
            print(cat_cols.values)
            pipe_cat = Pipeline(
            steps = [
                ("selector_cat", ColumnTransformer([("selector", "passthrough", cat_cols.values)], remainder = 'drop')),
                ('cat_imputer', SimpleImputer(strategy='most_frequent')),
                ("OrdinalEnc", OrdinalEncoder(handle_unknown = "use_encoded_value", unknown_value = -1))
            ])        
        else:
            pipe_cat = None
        
        if pipe_num is not None and pipe_cat is not None:
            prep_feat = FeatureUnion(
            transformer_list = [
                ('num_pipe', pipe_num),
                ('cat_pipe', pipe_cat)
                
            ], verbose = False)  
            print('Deu boas')
            print(dataframe[cat_cols])
            print(num_cols.values)
            print(cat_cols.values)
            print('oi')
            return (prep_feat, num_cols.values, cat_cols.values)

        if pipe_num is not None and pipe_cat is None:
            prep_feat = FeatureUnion(
            transformer_list = [
                ('num_pipe', pipe_num)
                
            ], verbose = False)
            
            return (prep_feat, num_cols.values, [])

        if pipe_num is None and pipe_cat is not None:
            prep_feat = FeatureUnion(
            transformer_list = [
                ('cat_pipe', pipe_cat)
                
            ], verbose = False) 
            
            return (prep_feat, [], cat_cols.values)



    def plot_dist(y_train, pred_proba_train, y_val, pred_proba_val):
        
        fig, axs = plt.subplots(nrows = 1, ncols = 2, figsize = (16,6))
        plt.subplots_adjust(left = None, right = None, top = None, bottom = None, wspace = 0.2, hspace = 0.4)
        
        vis = pd.DataFrame()
        vis['target'] = y_train
        vis['proba'] = pred_proba_train
        
        list_1 = vis[vis.target == 1].proba
        list_2 = vis[vis.target == 0].proba
        
        sns.distplot(list_1, kde = True, ax = axs[0], hist = True, bins = 100)
        sns.distplot(list_2, kde = True, ax = axs[0], hist = True, bins = 100)
        
        axs[0].set_title('train Thereshold Curve')
        
        
        
        vis = pd.DataFrame()
        vis['target'] = y_val
        vis['proba'] = pred_proba_val
        
        list_1 = vis[vis.target == 1].proba
        list_2 = vis[vis.target == 0].proba
        
        sns.distplot(list_1, kde = True, ax = axs[1], hist = True, bins = 100)
        sns.distplot(list_2, kde = True, ax = axs[1], hist = True, bins = 100)
        
        axs[1].set_title('valid Thereshold Curve')

        print('Saving fig')
        strFile = 'grafico_dist_base_0.png'
        while True:        
            if os.path.isfile(strFile):
                print('This file already exist, adding one')
                id = int(strFile[-5]) + 1
                strFile = strFile[:-5] + str(id) + strFile[-4:]
            else:
                print('Plot finished')
                break
        plt.savefig(strFile)

    def get_ir(df, target):
        valuer = df[target].value_counts().values.tolist()
        maxin = max(valuer)
        minin = min(valuer)

        return maxin/minin
    
    def transform_dat_to_csv(name_archive):
        # read flash.dat to a list of lists
        datContent = [i.strip().split() for i in open(name_archive).readlines()]

        # write it as a new CSV file
        with open(name_archive, "wb") as f:
            writer = csv.writer(f)
            writer.writerows(datContent)