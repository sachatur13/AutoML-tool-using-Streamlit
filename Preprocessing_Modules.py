import autogluon
from autogluon.tabular import TabularDataset,TabularPredictor
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time
import ppscore as pps

def get_predictive_power_score(dataset,target):

    predictive_power = pps.predictors(dataset,target)

    high_predictive_variables = predictive_power[['x','ppscore']][predictive_power['ppscore']>0]

    return predictive_power,high_predictive_variables

def drop_variables(dataset):


    categorical_variables = dataset.select_dtypes(include = 'O').columns
    
    high_frequency = [col for col in categorical_variables if dataset[col].nunique() > 10]


    return high_frequency

def feature_generator(dataset,selected_features):

    date_fields = dataset.select_dtypes(['datetime','datetime64']).columns

    for i in date_fields:

        for j in selected_features:
            
            if j == 'Year':
                dataset[i+'_Year'] = dataset[i].dt.year
            if j == 'Month':
                dataset[i+'_Month'] = dataset[i].dt.month
            if j == 'Quarter':
                dataset[i+'_Quarter'] = dataset[i].dt.quarter
            if j == 'Week of Year':
                dataset[i+'_WeekofYear'] = dataset[i].dt.weekofyear
            if j == 'Day of Week':
                dataset[i+'_DayofWeek'] = dataset[i].dt.dayofweek
            if j == 'Weekday Name':
                dataset[i+'_WeekdayName'] = dataset[i].dt.day_name()
    
    dataframe_with_generated_features = dataset.drop(date_fields,axis = 1)

    return dataframe_with_generated_features

def aggregated_data(group_col,agg_col,aggregation,dataset):

    if 'Sum' in aggregation:

        dataset['Total_per_'+group_col] = dataset.groupby(group_col)[agg_col].transform('sum')
        
    if 'Average' in aggregation:

        dataset['Average_per_'+group_col] = dataset.groupby(group_col)[agg_col].transform('mean')

    if 'Count' in aggregation:

        dataset['Count_per_'+group_col] = dataset.groupby(group_col).count()
        
    return dataset