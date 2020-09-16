import numpy as np
import pandas as pd
class BernoulliNaiveBayes():
    def __init__(self, smoothing=0):
        self.smoothing = smoothing
        
    def fit(self, dataframe, labels_column):
        """
        Calculo los parámetros del modelo
        """
        self.labels_column = labels_column
        self.labels = dataframe[labels_column].unique() #Me quedo con los nombres de las clases
        self.columns = dataframe.drop(columns=labels_column).columns.to_list() #Me quedo con los nombres de los predictores
        self.get_probas(dataframe)
        
    def get_probas(self, dataframe):
        """
        Calculo las log probabilidades a priori y likelihood.
        Uso log probabilidades y no probabilidades para hacer un mejor aprovechamiento de las posibilidades de representación numérica de los floats
        """
        self.log_prioris = {}
        self.log_likelihoods_1 = {}
        self.log_likelihoods_0 = {}
        for label in self.labels:
            subset = dataframe[dataframe[self.labels_column]==label] # Me quedo con los datos pertenecientes a solo una clase
            subset = subset.drop(columns=self.labels_column)
            self.log_prioris[label] = np.log(len(subset)/len(dataframe)) # Calculo las log probabilidades a priori
            alpha = self.smoothing
            proba = (subset.sum()+alpha)  / (len(subset)+alpha*2) #Calculo las probabilidades con smoothing
            self.log_likelihoods_1[label] = np.log( proba ) # Calculo las log probabilidades condicionales
            self.log_likelihoods_0[label] = np.log( 1-proba ) # Calculo las log probabilidades condicionales
            
    def predict_log_proba(self, dataframe):
        """
        Devuelve un dataframe con las log probabilidades de cada clase para cada observación
        """
        dataframe = dataframe.drop(columns=self.labels_column)
        output = {}
        for label in self.labels:
            output[label] = dataframe.mul(self.log_likelihoods_1[label]).sum(axis=1) + \
                (1-dataframe).mul(self.log_likelihoods_0[label]).sum(axis=1) + \
                self.log_prioris[label]
        return pd.DataFrame(output)
    
    def predict_proba(self, dataframe):
        """
        Devuelve un dataframe con las probabilidades de cada clase para cada observación
        """
        log_probas = np.exp(self.predict_log_proba(dataframe))
        probas = log_probas.div(log_probas.sum(axis=1), axis=0)
        return probas
    
    def predict(self,dataframe):
        """
        Devuelve una serie con la clase de mayor probabilidad para cada observación
        """
        log_probas = self.predict_log_proba(dataframe)
        clases = log_probas.idxmax(axis=1)
        return clases