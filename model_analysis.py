
import autogluon
from autogluon.tabular import TabularDataset,TabularPredictor
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression,LinearRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,f1_score,r2_score,mean_absolute_error
import ppscore as pps
#import modules.py as md
st.set_page_config(layout='wide')

st.header('Machine learning model training and testing app')
st.write('This app helps to train classification and regression(coming soon) in manual and automated mode. \nFor Manual classification tasks sklearn LogisticRegression and RandomForest are used in default settings. \n For automated model training [Autogluon](https://auto.gluon.ai/stable/index.html) library is leveraged.')


@st.cache()
def get_predictive_power_score(dataset,target):

    predictive_power = pps.predictors(dataset,target)

    high_predictive_variables = predictive_power[['x','ppscore']][predictive_power['ppscore']>0]

    return predictive_power,high_predictive_variables

@st.cache()
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


@st.cache()
def manual_model_regression(model_type,train_X,train_y,test_X,test_y):

    if model_type == 'Linear Regression':

        reg_model = LinearRegression()
                    
        reg_model.fit(train_X,train_y)    
                        
        score = r2_score(test_y,reg_model.predict(test_X))
    
    if model_type == 'Random Forest':

        reg_model = RandomForestRegressor()

        reg_model.fit(train_X,train_y)

        score = mean_absolute_error(test_y,reg_model.predict(test_X))

    return reg_model,np.round(score,2)
                        
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


st.header('Upload dataset')
dataset = st.file_uploader("",)

col1,col2 = st.beta_columns(2)


if dataset is not None:

    try:

#### Read dataset from upload
        input_dataset = pd.read_csv(dataset)

#### Change column names to lower case
        input_dataset.columns = input_dataset.columns.str.lower()

        st.success('File upload successful')

    except ValueError:
        st.error('Something went wrong while uploading the dataset')
else:
    st.markdown('### Dataset not found. Please upload a dataset first in CSV format')
  
if dataset is not None:

#### Identify integer, object and date type variables
    dataset_integer_variables = input_dataset.select_dtypes(include = ['int','int32','float','float32']).columns
    dataset_object_variables = input_dataset.select_dtypes(include = ['O']).columns
    dataset_date_variables = input_dataset.select_dtypes(include = ['datetime']).columns
    

### UI options for selecting operations

    st.header('Select operation')

    with st.beta_expander('Data type conversion'):
        
        selected_columns = st.multiselect('Select features for datatype conversion',input_dataset.columns)
        datatype_selection = st.selectbox('Select datatype to convert to',['','Integer','Object','Datetime'])

        if datatype_selection=="Datetime":
            
            try:
                
                for i in selected_columns:
                    
                    if i not in dataset_date_variables:
                       
                        input_dataset[i] = pd.to_datetime(input_dataset[i])
                
                    st.success('Data type conversion complete')

            except ValueError:
                st.error('Seems there is a problem with your variable selection')
    
        if datatype_selection=="Integer":
            
            try:
                
                for i in selected_columns:
                    
                    if i not in dataset_integer_variables:
                       
                        input_dataset[i] = input_dataset[i].astype('int')
                
                    st.success('Data type conversion complete')

            except ValueError:
                st.error('Seems there is a problem with your variable selection')
    
        if datatype_selection =="Object":
            
            try:
            
                for i in selected_columns:
                    
                    if i not in dataset_object_variables:
                        
                        input_dataset[i] = input_dataset[i].astype('O')
                    
                    st.success('Data type conversion complete')
            
            except ValueError:
                st.error('Seems there is a problem with your variable selection')


#### Dataset details section    
    if dataset is not None:
            
        with st.beta_expander('View dataset details'):
            
            st.markdown('#### Number of records: ')
            st.write(input_dataset.shape[0])
            
            st.markdown('#### Number of columns: ')
            st.write(input_dataset.shape[1])

            if st.checkbox('Show sample'):

                record_count = st.slider('Select number of records to show',1,20)
                
                st.write(input_dataset.head(record_count))

#### Data visualization section
        with st.beta_expander('Data Visualization'):

            st.markdown('#### Select visualization type')
            visual_type = st.radio('',('Univariate','Bi-variate'))

            if visual_type == 'Univariate':
                
                st.markdown('#### Select variables to plot')
                variables = st.selectbox('',input_dataset.columns)

                st.markdown('#### Adjust Figure width')
                fig_w = st.slider('',5,20,key='st_1')
                
                st.markdown('#### Adjust Figure height')
                fig_h = st.slider('',2,20,key='st_2')

                if variables in dataset_integer_variables:
                
                    fig,ax = plt.subplots(figsize =(fig_w,fig_h))
                    ax.hist(input_dataset[variables])
                    st.pyplot(fig)

                if variables in dataset_object_variables:
                    
                    height = st.selectbox('select x-axis',dataset_integer_variables)
                    fig,ax = plt.subplots(figsize =(fig_w,fig_h))
                    ax.bar(input_dataset[variables],input_dataset[height])
                    st.pyplot(fig)
        
            if visual_type == "Bi-variate":
                
                st.write('Only numeric variables supported. Updates coming soon')

                x = st.selectbox('Select variables to plot on x axis',dataset_integer_variables)
                
                y = st.selectbox('Select variables to plot on y axis',dataset_integer_variables)

                fig,ax = plt.subplots()
                ax.scatter(input_dataset[x],input_dataset[y])
                plt.xlabel(x)
                plt.ylabel(y)
                st.pyplot(fig)

    with st.beta_expander('Predictive Power'):

        target_var = st.selectbox('Select Target Variable',input_dataset.columns)

        predictive_p,high_predictive_variables = get_predictive_power_score(input_dataset,target_var)

        st.write('Variable ',high_predictive_variables['x'].unique(),' have highest predictability based on ppscore method')

        if st.checkbox('View full table'):
            st.dataframe(predictive_p)
        
        if st.checkbox('View as plot instead'):

            fig_w_pps = st.slider('Adjust Figure width',5,20,key='st_3')
            fig_h_pps = st.slider('Adjust Figure height',2,20,key='st_4')
        
            fig,ax = plt.subplots(figsize =(fig_w_pps,fig_h_pps))

            ax.barh(predictive_p['x'],predictive_p['ppscore'])
            plt.xlabel('Predictive Score')
            plt.ylabel('Variable')
            st.pyplot(fig)
        
        selected_features =  st.checkbox('Do you want to use these variables for next steps?')
        
        if selected_features==True:

            high_pred = list(high_predictive_variables['x'].unique())
            
            high_pred.append(target_var)

            updated_dataset_with_selected_features = input_dataset[high_pred]

            st.write(updated_dataset_with_selected_features)

    with st.beta_expander('Data preprocessing'):

        if selected_features==True:

            working_dataset = updated_dataset_with_selected_features
        
        else:

            working_dataset = input_dataset

        st.write('The preprocessing methods to be updated for better flow')

        col6,col7 = st.beta_columns(2)

        with col6:
            operations = st.multiselect('',['Remove High frequency columns','Encode categorical columns','Process Nulls'])
        
        with col7:
            for i in operations:
                #try:
                if i == 'Remove High frequency columns':
                    
                    high_frequency = drop_variables(working_dataset)

                    working_dataset = working_dataset.drop(high_frequency,axis = 1)
                    
                    dataset_integer_variables = working_dataset.select_dtypes(include = ['int','int32','float','float32']).columns
                    dataset_object_variables = working_dataset.select_dtypes(include = ['O']).columns
                
                if i == 'Encode categorical columns':

                    working_dataset = pd.get_dummies(working_dataset)               

                if i == 'Process Nulls':
                        
                    Null_columns = working_dataset.isnull().mean()

                    Null_process_method = st.multiselect('Select null imputation method: ',['Fill NA','Drop'])     

                    for j in Null_process_method:
                            
                        if j=="Drop":

                            working_dataset = working_dataset.dropna(thresh = 0.8,axis = 1)
                        
                        if j=="Fill NA":
                            
                            for k in dataset_integer_variables:
                                working_dataset[k] = working_dataset[k].fillna(np.mean(working_dataset[k]))

                            for k in dataset_object_variables:
                                working_dataset[k] = working_dataset[k].fillna(working_dataset[k].mode())     
        
        
        st.markdown('#### Generate date features: ')

        date_feature_generator = st.multiselect('',[None,'Year','Month','Quarter','Week of Year','Day of Week','Weekday Name'])

        if date_feature_generator is not None:

            working_dataset = feature_generator(working_dataset,date_feature_generator)

        st.write(working_dataset.head(2))
                #except:
                 #  st.error('Something went wrong during data preprocessing')
        st.write('Updates Coming soon')

    
    with st.beta_expander('Model Training'):

        modeling_dataset = working_dataset
        st.write(working_dataset)
        st.write('Select method for model training: ')

        training_method = st.radio('Manual Training',['Manual Training','Automated Model Training'])
        
        if training_method=='Manual Training':

            problem_type = st.selectbox('Select Problem Type',[None,'Classification','Regression'])

            col3,col4 = st.beta_columns(2)

            if problem_type == 'Regression':

                with col3:
                    
                    model_type = st.radio('Select training algorithm',['Linear Regression','Random Forest'])
                    
                    training_size = st.slider('Select Training set size',0.1,1.0,step=0.1,)
                    
                    test_size = np.round(1-training_size,2)

                with col4:

                    target = st.selectbox('Select Target variable (Integer only)',modeling_dataset.columns)

                    st.write('\n Training set  size (%) : ',training_size)
                    
                    st.write('\n Test set size (%) :',test_size)

                    train_X,test_X,train_y,test_y = train_test_split(modeling_dataset.drop(target,axis = 1),modeling_dataset[target],test_size=test_size,random_state = 42)
                
                if st.checkbox('Start Training'):

                    model,score = manual_model_regression(model_type,train_X,train_y,test_X,test_y)

                    if model_type == 'Linear Regression':

                        st.write('\n Linear Regression R2 Value: ',score)
                
                    if model_type == 'Random Forest':

                        st.write('\n Random Forest MAE Value: ',score)

            if problem_type =='Classification':

                with col3:
                
                    model_type = st.radio('Select training algorithm',['Logistic Regression','Random Forest'])

                    training_size = st.slider('Select Training set size',0.1,1.0,step=0.1,)
                    
                    test_size = np.round(1-training_size,2)

                with col4:
                    
                    target = st.selectbox('Select Target variable',modeling_dataset.columns)
                    target = target.lower()
                    
                    if st.checkbox('\n Convert target to object type?'):
                    
                        modeling_dataset[target] = modeling_dataset[target].astype('O')

                    st.write('\n Training set  size (%) : ',training_size)
                    
                    st.write('\n Test set size (%) :',test_size)

                    train_X,test_X,train_y,test_y = train_test_split(modeling_dataset.drop(target,axis = 1),modeling_dataset[target],test_size=test_size,random_state = 42)
                    
                if st.checkbox('Start Training'):

                    accuracy,f1_score,confusion_matrix,model = manual_model_training_classification(model_type,train_X,train_y,test_X,test_y)

                    st.write(model_type,' Accuracy: ',accuracy)
                    st.write(model_type,' F1 Score: ',f1_score)
                    st.table(confusion_matrix)
                        
                    if st.checkbox('Export Model'):
                        
                        import pickle

                        pickle.dump(model,open(model_type+'_export.sav','wb'))


        if training_method=='Automated Model Training':

            st.markdown('This method uses [Autogluon](https://auto.gluon.ai/stable/index.html) library for automated model training and comparison')
    
            target_variable = st.text_input("Input target variable")
            target_variable = target_variable.lower()
            
            if st.checkbox('Convert target to object type?'):

                modeling_dataset[target_variable] = modeling_dataset[target_variable].astype('O')          
            
            size_train = np.round(st.number_input('Training size'),2)
            size_test = np.round(1-(size_train),2)
            st.write('Training set size: ',size_train)
            st.write('Test set size: ',size_test)

            if size_train>0:

    #### Train and testing set creation

                test_data = modeling_dataset.sample(frac = size_test)
                train_data = modeling_dataset.sample(frac = size_train)
            
                if st.checkbox('Start model analysis'):
                    predictor,model_leaderboard = func_classification(train_data,test_data,target_variable)

                    st.markdown('## Resulting model leaderboard')
                    st.dataframe(model_leaderboard[['model','score_val']].sort_values('score_val',ascending = False))

#### Plot model output
                if st.checkbox('View plots'): 
                    fig,ax = plt.subplots()
                    ax.barh(model_leaderboard['model'],model_leaderboard['score_val'])
                    plt.xlabel('Validation score')
                    plt.ylabel('Model')
                    #plt.xlim(1)
                    st.write(fig)
                
                    trained_models = model_leaderboard['model']
