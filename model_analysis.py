
import autogluon
from autogluon.tabular import TabularDataset,TabularPredictor
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout='wide')

st.header('Auto model analysis app')
st.write('This app helps to identify the best performing model for a preprocessed dataset by leveraging [Autogluon](https://auto.gluon.ai/stable/index.html) library.')
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

def drop_variables(dataset):

    categorical_variables = dataset.select_dtypes(include = 'O').columns
    
    high_frequency = [col for col in categorical_variables if dataset[col].nunique() > 10]


    return high_frequency

st.sidebar.header('Upload dataset')
dataset = st.sidebar.file_uploader("",)

if dataset is not None:

#### Read dataset from upload
    dataset_source = pd.read_csv(dataset)

#### Change column names to lower case
    dataset_source.columns = dataset_source.columns.str.lower()

#### Identify integer, object and date type variables
    dataset_integer_variables = dataset_source.select_dtypes(include = ['int','int32','float','float32']).columns
    dataset_object_variables = dataset_source.select_dtypes(include = ['O']).columns
    datase_date_variables = dataset_source.select_dtypes(include = ['datetime']).columns
    
    st.header('Select operation')

#### Dataset details section    
    if dataset is not None and st.checkbox('Show dataset details'):
        
        st.header('Dataset details: ')
        st.write('Number of records: ',dataset_source.shape[0])
        st.write('Number of columns: ',dataset_source.shape[1])

        if st.checkbox('Show sample'):
            record_count = st.slider('Select number of records to show',1,20)
            st.table(dataset_source.head(record_count))

#### Data visualization section
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
        
        if visual_type == "Bi-variate":
            st.write('Only numeric variables supported. Updates coming soon')
            x = st.selectbox('Select variables to plot on x axis',dataset_integer_variables)
            y = st.selectbox('Select variables to plot on y axis',dataset_integer_variables)

            fig,ax = plt.subplots()
            ax.scatter(dataset_source[x],dataset_source[y])
            plt.xlabel(x)
            plt.ylabel(y)
            st.pyplot(fig)

#### Model training section    
    if st.checkbox('Train models'):
        col1,col2 = st.beta_columns(2)
    
        with col1:
            target_variable = st.text_input("Input target variable")
            target_variable = target_variable.lower()
            if st.checkbox('Convert target to object type?'):
                dataset_source[target_variable] = dataset_source[target_variable].astype('O')          

        with col2:
            size_train = np.round(st.number_input('Training size'),2)
            size_test = np.round(1-(size_train),2)
            st.write('Training set size: ',size_train)
            st.write('Test set size: ',size_test)

        if size_train>0:

#### Drop high frequency variables
            high_frequency = drop_variables(dataset_source)
            dataset_source = dataset_source.drop(high_frequency,axis = 1)

#### Train and testing set creation
            test_data = dataset_source.sample(frac = size_test)
            train_data = dataset_source.sample(frac = size_train)
            
            if st.checkbox('Start model analysis'):
                predictor,model_leaderboard = func_classification(train_data,test_data,target_variable)

                st.write('Problem type: ',predictor.problem_type)
                st.write('Features identified: ',predictor.feature_metadata)
                
                with st.beta_expander('View results as table'):
                    st.table(model_leaderboard[['model','score_test','score_val']])

#### Plot model output
                with st.beta_expander('View plots'): 
                    fig,ax = plt.subplots()
                    ax.barh(model_leaderboard['model'],model_leaderboard['score_val'])
                    plt.xlabel('Evaluated Model')
                    plt.ylabel('Validation score')
                    st.pyplot(fig)
                
                trained_models = model_leaderboard['model']

                model_name = st.sidebar.selectbox('Trained models',trained_models)
                

else:
    st.write('Dataset not found. Please upload a dataset first in CSV format')
