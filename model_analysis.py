
import autogluon
from autogluon.tabular import TabularDataset,TabularPredictor
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time

st.set_page_config(layout='wide')

st.header('Auto model analysis app')
st.write('This app helps to identify the best performing model for a preprocessed dataset by leveraging [Autogluon](https://auto.gluon.ai/stable/index.html) library.')

@st.cache()
def func_train_test_split(dataset,target_variable,size_train,size_test):

    X = dataset.drop(target_variable,axis = 1)
    y = dataset [target_variable].astype('O')

    if size_test<1:

        train_X,test_X,train_y,test_y = train_test_split(X,y,test_size = size_test,random_state = 42)

    return train_X,test_X,train_y,test_y
@st.cache()
def func_classification(train_data,test_data,label):
    
    predictor = TabularPredictor(label = label,eval_metric = 'f1').fit(train_data)
    
    #test_data = test_data.drop(label,axis = 1)

    model_leaderboard = predictor.leaderboard(test_data,silent = True)  
    
    return predictor,model_leaderboard


def drop_variables(dataset):

    categorical_variables = dataset.select_dtypes(include = 'O').columns
    
    high_frequency = [col for col in categorical_variables if dataset[col].nunique() > 10]


    return high_frequency


st.header('Upload dataset')
dataset = st.file_uploader("",)

col1,col2 = st.beta_columns(2)


if dataset is not None:

#### Read dataset from upload
    dataset_source = pd.read_csv(dataset)

#### Change column names to lower case
    dataset_source.columns = dataset_source.columns.str.lower()
    

        
else:
    st.write('Dataset not found. Please upload a dataset first in CSV format')
    
  
if dataset is not None:

    st.header('Select operation')

    with st.beta_expander('Data preprocessing'):
        col6,col7 = st.beta_columns(2)

        with col6:
            operations = st.multiselect('',['Remove High frequency columns','Encode categorical columns'])
        
        with col7:
            for i in operations:
                try:
                    if i == 'Remove High frequency columns':
                    
                        high_frequency = drop_variables(dataset_source)
                        dataset_source = dataset_source.drop(high_frequency,axis = 1)
                
                    if i == 'Encode categorical columns':

                        dataset_source = pd.get_dummies(dataset_source)                    

                    
                except:
                    st.error('Something went wrong during data preprocessing')
        st.write('Updates Coming soon')

    with st.beta_expander('Data type conversion'):
        
        selected_columns = st.multiselect('Select features for datatype conversion',dataset_source.columns)
        datatype_selection = st.selectbox('Select datatype to convert to',['','Integer','Object'])

        if datatype_selection=="Integer":
            try:
                for i in selected_columns:
                    if i not in dataset_integer_variables:
                        dataset_source[i] = dataset_source[i].astype('int')
                
                    st.success('Data type conversion complete')
            except ValueError:
                st.error('Seems there is a problem with your variable selection')
    
        if datatype_selection =="Object":
            try:
                for i in selected_columns:
                    if i not in dataset_object_variables:
                        dataset_source[i] = dataset_source[i].astype('O')
                    st.success('Data type conversion complete')
            except ValueError:
                st.error('Seems there is a problem with your variable selection')

#### Identify integer, object and date type variables
        dataset_integer_variables = dataset_source.select_dtypes(include = ['int','int32','float','float32']).columns
        dataset_object_variables = dataset_source.select_dtypes(include = ['O']).columns
        datase_date_variables = dataset_source.select_dtypes(include = ['datetime']).columns
    

#### Dataset details section    
    if dataset is not None:
            
        with st.beta_expander('View dataset details'):
            st.header('Dataset details: ')
            st.write('Number of records: ',dataset_source.shape[0])
            st.write('Number of columns: ',dataset_source.shape[1])

            if st.checkbox('Show sample'):
                record_count = st.slider('Select number of records to show',1,20)
                st.write(dataset_source.head(record_count))

#### Data visualization section
        with st.beta_expander('Data Visualization'):

            visual_type = st.radio('Visualize data',('Univariate','Bi-variate'))

            if visual_type == 'Univariate':
                
                variables = st.selectbox('Select variables to plot',dataset_source.columns)

                if variables in dataset_integer_variables:
                    plt.figure(figsize = (10,10))
                    fig,ax = plt.subplots()
                    ax.hist(dataset_source[variables])
                    st.write(fig)

                if variables in dataset_object_variables:
                    height = st.selectbox('select x-axis',dataset_integer_variables)
                    fig,ax = plt.subplots()
                    ax.bar(dataset_source[variables],dataset_source[height])
                    st.write(fig)
        
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
    with st.beta_expander('Model Training'):

        st.write('Select method for model training: ')
        training_method = st.radio('Manual Training',['Manual Training','Automated Model Training'])
        
        if training_method=='Manual Training':

            problem_type = st.selectbox('Select Problem Type',[None,'Classification','Regression'])

            col3,col4 = st.beta_columns(2)

            if problem_type =='Classification':
                with col3:
                    model_type = st.radio('Select training algorithm',['Logistic Regression','Random Forest'])

                    training_size = st.slider('Select Training set size',0.1,1.0,step=0.1,)
                    test_size = np.round(1-training_size,2)

                with col4:
                    target = st.selectbox('Select Target variable',dataset_source.columns)
                    target = target.lower()
                    if st.checkbox('\n Convert target to object type?'):
                        dataset_source[target_variable] = dataset_source[target_variable].astype('O')

                    st.write('\n Training set  size (%) : ',training_size)
                    st.write('\n Test set size (%) :',test_size)          
           

            st.write(problem_type)

        if training_method=='Automated Model Training':

            st.markdown('This method uses [Autogluon](https://auto.gluon.ai/stable/index.html) library for automated model training    ')
    
            target_variable = st.text_input("Input target variable")
            target_variable = target_variable.lower()
            
            if st.checkbox('Convert target to object type?'):
                dataset_source[target_variable] = dataset_source[target_variable].astype('O')          
            
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

                    st.markdown('Problem type: ',predictor.problem_type)
                    st.markdown('Features identified: ',predictor.feature_metadata)
                
                if st.checkbox('View results as table'):
                    st.write(model_leaderboard[['model','score_test','score_val']])

#### Plot model output
                if st.checkbox('View plots'): 
                    fig,ax = plt.subplots()
                    ax.barh(model_leaderboard['model'],model_leaderboard['score_val'])
                    plt.xlabel('Validation score')
                    plt.ylabel('Model')
                    #plt.xlim(1)
                    st.write(fig)
                
                    trained_models = model_leaderboard['model']

                    #with st.beta_expander('Make predictions with unlabelded input data or Test the models with your test data'):
                     #   st.header('Upload test dataset')
                      #  test_dataset = st.file_uploader("",key='Fileuploader01')
                       # test_source = pd.read_csv(test_dataset)
                        #test_source.columns = test_source.columns.str.lower()
                        
                        #if st.checkbox('Test models'):
                         #   tested_models = predictor.leaderboard(test_source,silent = True)
                          #  st.table(tested_models[['model','score_test','score_val']])

                        ##if st.checkbox('Make predictions'):
                        
                          #  st.write('This uses the best performing model. Dynamic model selection coming soon')
                           # st.progress(100)    
                                              
                            #test_source.drop([target_variable],axis = 1,inplace = True)
                            #y_pred = predictor.predict(test_source)
                            #test_source['Predicted'] = y_pred 
                            #st.write(test_source)


                
