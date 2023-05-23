import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

import numpy as np
from sklearn.ensemble import AdaBoostClassifier


class AdaCostClassifier(AdaBoostClassifier): 

    def __init__(self, estimator=None, n_estimators=150, learning_rate=0.5, 
                 FNcost='auto', FPcost=1, algorithm='SAMME.R', random_state=None):
        super(AdaBoostClassifier, self).__init__(
            estimator=estimator,n_estimators=n_estimators,
            learning_rate=learning_rate, random_state=random_state)

        self.FPcost = FPcost
        self.FNcost = FNcost
        self.algorithm = algorithm
    
    def _boost_real(self, iboost, X, y, sample_weight, random_state):
        """Implement a single boost using the SAMME.R real algorithm."""
        estimator = self._make_estimator(random_state=random_state)
        estimator.fit(X, y, sample_weight=sample_weight)
        y_predict_proba = estimator.predict_proba(X)

        if iboost == 0:
            self.classes_ = getattr(estimator, 'classes_', None)
            self.n_classes_ = len(self.classes_)
        y_predict = self.classes_.take(np.argmax(y_predict_proba, axis=1), axis=0)
        incorrect = y_predict != y
        estimator_error = np.mean(np.average(incorrect, weights=sample_weight, axis=0))
        if estimator_error <= 0:
            return sample_weight, 1., 0.

        n_classes = self.n_classes_
        classes = self.classes_
        y_codes = np.array([-1. / (n_classes - 1), 1.])
        y_coding = y_codes.take(classes == y[:, np.newaxis])
        proba = y_predict_proba  # alias for readability
        

 
        np.clip(proba, np.finfo(proba.dtype).eps, None, out=proba)
        
        estimator_weight = (-1. * self.learning_rate * ((n_classes - 1.) / n_classes)
                            * (y_coding * np.log(y_predict_proba)).sum(axis=1))
        
        
        if not iboost == self.n_estimators - 1:
            criteria = ((sample_weight > 0) | (estimator_weight < 0))
           
            sample_weight *= np.exp(estimator_weight * criteria * self._beta(y, y_predict))  
        return sample_weight, 1., estimator_error

  
    def _beta(self, y, y_hat):
        res = []
        ratio = sum(y==0) / sum(y==1)
        if self.FNcost == 'auto':
            self.FNcost = ratio
        
        for i in zip(y, y_hat):
           
            if  i[0] == 1 and i[1] == 0:
                res.append(self.FNcost)  
            
            elif i[0] == 0 and i[1] == 1:
                res.append(self.FPcost)  
            
            else:
                res.append(1)
        return np.array(res)