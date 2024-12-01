import numpy as np
import pandas as pd
import random
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.metrics import mean_absolute_error,mean_squared_error
from sklearn.ensemble import RandomForestRegressor

class Objectif_Function:
    

    def __init__(self, X,y):
        self.X = X
        self.y = y

    def get_objectif_values(self,individu ):
        return [self.f1(individu),self.f2(individu) ]
    
    def f1(self, individu):
        """ 
            Cette methode permet de calculer l erreur de classification pour un individu
        """
        df_split = self.split_data(individu)
        return self.train_model(df_split)

    def f2(self,individu):    
        return np.sum(individu)

    def classification_error(self, y_true , y_predict):
        """ 
        Cette methode permet de calculer l erreur de classification 
        en comparant les vrais labels avec les labels predits
        """
        wrongly_predicted = np.sum(y_true != y_predict)
        number_all_instance = len(y_true)
        cla_error = wrongly_predicted /number_all_instance
        return cla_error
    
    def split_data(self, individu):
        #X = df.iloc[:,:-1]
        #y = df[df.columns[-1]]
        #X_train, X_test, y_train, y_test = train_test_split(self.X, self.y , test_size = 0.3,random_state=0)
        data = {
            "train":{"X": self.X.iloc[:, individu.astype(bool)], 
                     "y": self.y},#Permet de recuperer uniquement les colonnes selectionnees

            "test":{"X": self.X.iloc[:, individu.astype(bool)], 
                    "y": self.y}
        }
        return data

    
    def train_model(self, data):
        """ Cette methode est utilisee pour entrainer le modèle KNN avec K = 3, 
                 utilisant une K-Fold avec K=3 selon l'article """
                 
        X = data["train"]["X"]
        y = data["train"]["y"]
        #print("X",X.shape)
        #print("y",y.shape)
        kf = KFold(n_splits=3, shuffle=True , random_state=64)            ## Dans l article, K = 3
        class_error = []
    
        for train_idex,test_idex in kf.split(X):
            X_train, X_test = X.iloc[train_idex], X.iloc[test_idex]
            y_train, y_test = y.iloc[train_idex], y.iloc[test_idex]
            knn = KNeighborsClassifier(n_neighbors=3)   # Dans l article, n_neighbors=3
            knn.fit(X_train, y_train)
            y_predict = knn.predict(X_test)
            class_error.append(self.classification_error(y_test, y_predict))
        
        return np.mean(np.array(class_error))*100
  