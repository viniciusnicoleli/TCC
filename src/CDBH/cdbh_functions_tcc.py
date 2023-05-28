import math
import operator as op
from sklearn.cluster import KMeans
import math
from scipy.spatial import KDTree
from random import choices, choice
import pandas as pd
import numpy as np

class cdbh_functions():
    def __init__(self, dataframe : pd.DataFrame, df_train: str):
        self.dataframe = dataframe
        self.target = df_train

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
            i = choices(X_train_mino.index, weights=X_train_mino.prob, k=1)
            vector_i = X_train_mino[X_train_mino.index == i[0]].drop(['cluster', 'density', 'prob'], axis=1)
        
            distances, index = temp_kdtree.query(vector_i, k_s)  
            
            index = index.tolist()[0]
            
            j = choice(index)
            rand = choice(np.linspace(0, 1, 100))
            vector_j = X_train_mino.iloc[j]
            
            sub_synt = list(map(op.sub, vector_i.values.flatten().tolist(), vector_j.values.flatten().tolist()))
        
            sub_synt = list(np.asarray(sub_synt)*rand)    
            
            my_list = [vector_i.values.flatten().tolist(),sub_synt]
            
            data_synthetic = list(map(sum, zip(*my_list)))
            
            X_train_mino_out.loc[-1] = data_synthetic
        
            X_train_mino_out.index = X_train_mino_out.index + 1    
            
            IR = df_train[df_train['Class'] == 0].drop(['Class'], axis=1).shape[0] / X_train_mino_out.shape[0]
        
        X_train_majo = df_train[df_train['Class'] == 0]
        
        
        k_s, X_train_majo['cluster'] = self.funcao_return_cluster(X_train_majo)
        temp = self.funcao_return_density(X_train_majo)
        
        X_train_majo = pd.merge(X_train_majo, temp, left_index=True, right_index=True)
        
        X_train_majo['prob'] = X_train_majo['density'].apply(lambda x: x/X_train_majo['density'].sum())
        
        while  X_train_majo.shape[0] > X_train_mino_out.shape[0]:
            i = choices(X_train_majo.index, weights=X_train_majo.prob, k=1)
            vector_i = X_train_majo[X_train_majo.index == i[0]].index.values
        
            X_train_majo = X_train_majo.drop(vector_i, axis=0)
            
        
        X_train_majo_out = X_train_majo.drop(['cluster', 'density', 'prob'], axis=1).copy()
        
        

        X_train_out = pd.concat([X_train_mino_out, X_train_majo_out], axis=0)
        
        return (X_train_out)        