## Feature engineering
## predictive power score
##Drop features
## Finding best model


import lazypredict 
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
from lazypredict.Supervised import LazyClassifier,LazyRegressor
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt

def func_train_test_split(dataset,target_variable,size_train,size_test):

    X = dataset.drop(target_variable,axis = 1)
    y = dataset [target_variable].astype('O')

    if size_test<1:

        train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = size_test,random_state = 42)

    return train_X,test_X,train_y,test_y

def func_classification(train_X,test_X,train_y,test_y):
    
    classifier = lazypredict.Supervised.LazyRegressor(verbose = 1,predictions = True)
    model,predictions = classifier.fit(train_X,test_X,train_y,test_y)    
    model = model.reset_index()

    return model,predictions

def drop_variables(dataset):

    categorical_variables = dataset.select_dtypes(include = 'O').columns
    
    high_frequency = [col for col in categorical_variables if dataset[col].nunique() > 10]


    return high_frequency

st.sidebar.header('Upload dataset')
dataset = st.sidebar.file_uploader("",)

if dataset is not None:

    dataset_source = pd.read_csv(dataset)
    dataset_source.columns = dataset_source.columns.str.lower()
    dataset_integer_variables = dataset_source.select_dtypes(include = ['int','int32','float','float32']).columns
    dataset_object_variables = dataset_source.select_dtypes(include = ['O']).columns
    datase_date_variables = dataset_source.select_dtypes(include = ['datetime']).columns
    
    st.header('Select operation')
    if dataset is not None and st.checkbox('Show dataset details'):
        st.header('Dataset details: ')
        st.write('Number of records: ',dataset_source.shape[0])
        st.write('Number of columns: ',dataset_source.shape[1])
        if st.checkbox('Show sample'):
            record_count = st.slider('Select number of records to show',1,20)
            st.table(dataset_source.head(record_count))

    
    
    if st.checkbox('Data visualization'):
        visual_type = st.radio('Visualize data',('Univariate','Bi-variate'))

        if visual_type == 'Univariate':
        
            variables = st.selectbox('Select variables to plot',dataset_source.columns)
            if variables in dataset_integer_variables:
                fig,ax = plt.subplots()
                ax.hist(dataset_source[variables])
                st.pyplot(fig)
            if variables in dataset_object_variables:
                height = st.selectbox('select x-axis',dataset_integer_variables)
                fig,ax = plt.subplots()
                ax.bar(dataset_source[variables],dataset_source[height])
                st.pyplot(fig)
    
    if st.checkbox('Train models'):
        col1,col2 = st.beta_columns(2)

    
        with col1:
            target_variable = st.text_input("Input target variable")
            target_variable = target_variable.lower()
            

        with col2:
            size_train = np.round(st.number_input('Training size'),2)
            size_test = np.round(1-(size_train),2)
            st.write('Training set size: ',size_train)
            st.write('Test set size: ',size_test)

        if size_train>0:
            high_frequency = drop_variables(dataset_source)

            dataset_source = dataset_source.drop(high_frequency,axis = 1)

            train_X,test_X,train_y,test_y = func_train_test_split(dataset_source,target_variable,size_train,size_test)

            if st.checkbox('Analyze models'):
                model,predictions = func_classification(train_X,test_X,train_y,test_y)
                st.write(model)
                selection = st.sidebar.selectbox('Select metric to visualize',['R-Squared','RMSE'])

            if st.sidebar.checkbox('Plot metrics'):
                plt.figure(figsize=(20,12))
                fig,ax = plt.subplots()
                ax.scatter(model[selection],model['Model'])
                st.pyplot(fig)


else:
    st.write('Dataset not found')
