# Warnings
import warnings
warnings.filterwarnings('ignore')

# Importing utilities
import os, sys
sys.path.insert(0, os.path.abspath(".."))
from utilidades.calibration import utilities as ult

# Basic imports
import numpy as np
import pandas as pd
import math
import random
import operator as op

# Model.
from sklearn.preprocessing import StandardScaler
from clusteval import clusteval
from scipy.spatial import KDTree
from sklearn.cluster import KMeans



class tcc_indices_to_resample():
    def __init__(self, dataframe : pd.DataFrame, target: str, indices = None):
        self.dataframe = dataframe
        self.target = target
        self.indices = indices
    
    def fit(self):
        temp = self.dataframe.copy()
        temp = temp[temp[self.target] == 1]
        
        X, y = ult.splitxy(temp, self.target)

        scaler = StandardScaler()
        X_ = scaler.fit_transform(X)
        X_ = pd.DataFrame(X_, columns = X.columns, index = X.index)
        
        ce = clusteval(cluster='agglomerative', evaluate='silhouette')
        X_.replace([np.inf, -np.inf], np.nan, inplace=True)
        X_.dropna(inplace=True)

        results = ce.fit(X_)
        X_['clusters'] = results['labx']
        
        clusters = pd.DataFrame(X_['clusters'].value_counts(normalize = True))
        
        #clusters_id = clusters[clusters.cumsum() <= 0.8].dropna().index.values
        clusters_id = clusters.index.values
        
        self.dataframe = pd.merge(self.dataframe, X_[['clusters']], left_index=True, right_index=True, how = 'left')
        
        self.dataframe['clusters'] = self.dataframe['clusters'].fillna(-1)
        
        idx_clusters = []
        for cluster in clusters_id:
            den = self.dataframe[self.dataframe['clusters'] == cluster].shape[0]
            
            
            self.dataframe['temp_y'] = np.where(self.dataframe['clusters'] == cluster, 1, 0)
            temp = self.dataframe.drop([self.target, 'clusters'], axis=1)
            
            X, y = ult.splitxy(temp, 'temp_y')
            scaler = StandardScaler()
            
            X_ = scaler.fit_transform(X)
            X_ = pd.DataFrame(X_, columns = X.columns, index = X.index)
            
            temp = pd.merge(X_, y, left_index=True, right_index=True, how = 'inner')
            
            temp1 = KDTree(temp.drop(['temp_y'],axis=1))
            
            lst_end = []
            for i_ in (2, 5, 6, 8, 10, 15):
                lst = [] 
                lst_idx = []
                for i in temp[temp['temp_y'] == 1].drop(['temp_y'], axis=1).index:
                    
                    index = temp.drop(['temp_y'], axis=1)[temp.index == i].values.flatten().tolist() #temp.drop(['temp_y'], 1).iloc[i].tolist()
                    
                    idx = temp1.query_ball_point(index,r = i_)
                    
                    x = temp.iloc[idx]['temp_y'].sum()/len(temp.iloc[idx]['temp_y']) 
                    lst.append(x)
                    lst_idx = lst_idx + idx
                    lst_idx = list(dict.fromkeys(lst_idx))
                    
                idn = pd.DataFrame(lst, columns = ['ratio'])
                
                end = idn[idn['ratio'] < 1].shape[0] / den
                
                lst_end.append([lst_idx, end])
                
                to_go = []
                for item in lst_end:
                    if to_go == []:
                        to_go = item
                    else:
                        if abs(item[1] - 0.7) < abs(to_go[1] - 0.7):
                            to_go = item
                        else:
                            pass 
                        
                idx_cluster = to_go[0]
                idx_clusters = idx_clusters + idx_cluster
                idx_clusters = list(dict.fromkeys(idx_clusters))     
        self.indices = idx_clusters

    def funcao_return_cluster(self,dataframe : pd.DataFrame) -> pd.Series:
        k_s = round(math.sqrt(dataframe.shape[0]),0)
        k_s = int(k_s)  
        
        k_means = KMeans(n_clusters=k_s, random_state=42)
        k_means.fit(dataframe) 
        
        lista_output = k_means.predict(dataframe)
        return (k_s, lista_output)
    
    def funcao_return_density(self,dataframe : pd.DataFrame):
        final_list = pd.DataFrame([], columns = ['density'])
        clusters = dataframe.cluster.value_counts().index
        
        for cluster in clusters:
            
            temp_dataframe = dataframe[dataframe['cluster'] == cluster]
            final_point = temp_dataframe.shape[0]
            dataframe_densidade_index = pd.DataFrame(range(0, final_point), columns = ['density'], index = temp_dataframe.index)
            dataframe_densidade_index['density'] = 0
            
            temp_kdtree = KDTree(temp_dataframe)
            
            
            if temp_dataframe.shape[0] == 1:
                pass
            else:
                for point in temp_dataframe.index:
                    
                    vector_i = temp_dataframe[temp_dataframe.index == point].values.flatten().tolist()
                
                    distances, index = temp_kdtree.query(vector_i, [2])
                    
                    dataframe_densidade_index.iloc[index] = dataframe_densidade_index.iloc[index] + 1
                
                final_list = pd.concat([final_list, dataframe_densidade_index], axis=0)
            
        return final_list
    
    def cdbh(self,df_train):
        X_train_mino = df_train[df_train['Class'] == 1]
        X_train_mino['cluster'] = self.funcao_return_cluster(X_train_mino)[1]
        k_s = self.funcao_return_cluster(X_train_mino)[0]
        temp = self.funcao_return_density(X_train_mino)
        X_train_mino = pd.merge(X_train_mino, temp, left_index=True, right_index=True)
        X_train_mino.head()        
        
        X_train_mino['prob'] = X_train_mino['density'].apply(lambda x: x/X_train_mino['density'].sum())
        
        X_train_mino_out = X_train_mino.drop(['cluster', 'density', 'prob'], axis=1).copy()
        temp_kdtree = KDTree(X_train_mino.drop(['cluster', 'density', 'prob'], axis=1))
        IR = df_train[df_train['Class'] == 0].drop(['Class'], axis=1).shape[0] / X_train_mino.shape[0]
        while IR > 2:
            i = random.choices(X_train_mino.index, weights=X_train_mino.prob, k=1)
            vector_i = X_train_mino[X_train_mino.index == i[0]].drop(['cluster', 'density', 'prob'], axis=1)
        
            distances, index = temp_kdtree.query(vector_i, k_s)  
            
            index = index.tolist()[0]
            
            j = random.choice(index)
            rand = random.choice(np.linspace(0, 1, 100))
            vector_j = X_train_mino.iloc[j]
            
            sub_synt = list(map(op.sub, vector_i.values.flatten().tolist(), vector_j.values.flatten().tolist()))
        
            sub_synt = list(np.asarray(sub_synt)*rand)    
            
            my_list = [vector_i.values.flatten().tolist(),sub_synt]
            
            data_synthetic = list(map(sum, zip(*my_list)))
            
            X_train_mino_out.loc[-1] = data_synthetic
        
            X_train_mino_out.index = X_train_mino_out.index + 1    
            
            IR = df_train[df_train['Class'] == 0].drop(['Class'], 1).shape[0] / X_train_mino_out.shape[0]
        
        X_train_majo = df_train[df_train['Class'] == 0]
        
        
        k_s, X_train_majo['cluster'] = self.funcao_return_cluster(X_train_majo)
        temp = self.funcao_return_density(X_train_majo)
        
        X_train_majo = pd.merge(X_train_majo, temp, left_index=True, right_index=True)
        
        X_train_majo['prob'] = X_train_majo['density'].apply(lambda x: x/X_train_majo['density'].sum())
        
        while  X_train_majo.shape[0] > X_train_mino_out.shape[0]:
            i = random.choices(X_train_majo.index, weights=X_train_majo.prob, k=1)
            vector_i = X_train_majo[X_train_majo.index == i[0]].index.values
        
            X_train_majo = X_train_majo.drop(vector_i, axis=0)
            
        
        X_train_majo_out = X_train_majo.drop(['cluster', 'density', 'prob'], axis=1).copy()
        
        

        X_train_out = pd.concat([X_train_mino_out, X_train_majo_out], 0)
        
        return (X_train_out)        