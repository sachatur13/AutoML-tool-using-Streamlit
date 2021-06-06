import autogluon
from autogluon.tabular import TabularDataset,TabularPredictor
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import time
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,f1_score,r2_score,mean_absolute_error
import ppscore as pps

def manual_model_training_classification(model_type,train_X,train_y,test_X,test_y):

    if model_type == 'Logistic Regression':
                        
        model = LogisticRegression()
                        
        model.fit(train_X,train_y)
                 
        accuracy = np.round(model.score(test_X,test_y),2)*100

        f1_score_value = np.round(f1_score(test_y,model.predict(test_X)),2)*100
                    
        confusion_matrix_return = confusion_matrix(test_y,model.predict(test_X))
                
    if model_type == 'Random Forest':

        model = RandomForestClassifier()

        model.fit(train_X,train_y)

        accuracy = np.round(model.score(test_X,test_y),2)*100

        f1_score_value = np.round(f1_score(test_y,model.predict(test_X)),2)*100
                        
        confusion_matrix_return = confusion_matrix(test_y,model.predict(test_X))

    return accuracy,f1_score_value,confusion_matrix_return,model


def manual_model_training_regression(model_type,train_X,train_y,test_X,test_y):

    if model_type == 'Linear Regression':

        reg_model = LinearRegression()
                    
        reg_model.fit(train_X,train_y)    
                        
        score = r2_score(test_y,reg_model.predict(test_X))
    
    if model_type == 'Random Forest':

        reg_model = RandomForestRegressor()

        reg_model.fit(train_X,train_y)

        score = mean_absolute_error(test_y,reg_model.predict(test_X))

    return reg_model,np.round(score,2)
                        
def func_train_test_split(dataset,target_variable,size_train,size_test):

    X = dataset.drop(target_variable,axis = 1)
    y = dataset [target_variable].astype('O')

    if size_test<1:

        train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = size_test,random_state = 42)

    return train_X,test_X,train_y,test_y

def func_classification(train_data,test_data,label):
    
    predictor = TabularPredictor(label = label,eval_metric = 'f1').fit(train_data)
    
    #test_data = test_data.drop(label,axis = 1)

    model_leaderboard = predictor.leaderboard(test_data,silent = True)  
    
    return predictor,model_leaderboard

