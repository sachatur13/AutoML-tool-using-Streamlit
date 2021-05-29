
import autogluon
from autogluon.tabular import TabularDataset,TabularPredictor
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,RandomForestRegressor
from sklearn.metrics import classification_report,confusion_matrix,f1_score

st.set_page_config(layout='wide')

st.header('Machine learning model training and testing app')
st.write('This app helps to train classification and regression(coming soon) in manual and automated mode. \nFor Manual classification tasks sklearn LogisticRegression and RandomForest are used in default settings. \n For automated model training [Autogluon](https://auto.gluon.ai/stable/index.html) library is leveraged.')


@st.cache

def manual_model_training(model_type,train_X,train_y,test_X,test_y):

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

    try:

#### Read dataset from upload
        dataset_source = pd.read_csv(dataset)

#### Change column names to lower case
        dataset_source.columns = dataset_source.columns.str.lower()

        st.success('File upload successful')

    except ValueError:
        st.error('Something went wrong while uploading the dataset')
else:
    st.write('Dataset not found. Please upload a dataset first in CSV format')
  
if dataset is not None:

#### Identify integer, object and date type variables
    dataset_integer_variables = dataset_source.select_dtypes(include = ['int','int32','float','float32']).columns
    dataset_object_variables = dataset_source.select_dtypes(include = ['O']).columns
    datase_date_variables = dataset_source.select_dtypes(include = ['datetime']).columns
    

### UI options for selecting operations

    st.header('Select operation')

    with st.beta_expander('Data preprocessing'):
        st.write('The preprocessing methods to be updated for better flow')

        col6,col7 = st.beta_columns(2)

        with col6:
            operations = st.multiselect('',['Remove High frequency columns','Encode categorical columns','Process Nulls'])
        
        with col7:
            for i in operations:
                #try:
                if i == 'Remove High frequency columns':
                    
                    high_frequency = drop_variables(dataset_source)

                    dataset_source = dataset_source.drop(high_frequency,axis = 1)
                    dataset_integer_variables = dataset_source.select_dtypes(include = ['int','int32','float','float32']).columns
                    dataset_object_variables = dataset_source.select_dtypes(include = ['O']).columns
                
                if i == 'Encode categorical columns':

                    dataset_source = pd.get_dummies(dataset_source)               

                if i == 'Process Nulls':
                        
                    Null_columns = dataset_source.isnull().mean()

                    Null_process_method = st.multiselect('Select null imputation method: ',['Fill NA','Drop'])     

                    for j in Null_process_method:
                            
                        if j=="Drop":

                            dataset_source = dataset_source.dropna(thresh = 0.8,axis = 1)
                        
                        if j=="Fill NA":
                            
                            for k in dataset_integer_variables:
                                dataset_source[k] = dataset_source[k].fillna(np.mean(dataset_source[k]))

                            for k in dataset_object_variables:
                                dataset_source[k] = dataset_source[k].fillna(dataset_source[k].mode())     
                    
                #except:
                 #  st.error('Something went wrong during data preprocessing')
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


#### Dataset details section    
    if dataset is not None:
            
        with st.beta_expander('View dataset details'):
            
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
                    
                        dataset_source[target] = dataset_source[target].astype('O')

                    st.write('\n Training set  size (%) : ',training_size)
                    
                    st.write('\n Test set size (%) :',test_size)

                    train_X,test_X,train_y,test_y = train_test_split(dataset_source.drop(target,axis = 1),dataset_source[target],test_size=test_size,random_state = 42)
                    
                if st.checkbox('Start Training'):

                    accuracy,f1_score,confusion_matrix,model = manual_model_training(model_type,train_X,train_y,test_X,test_y)

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

                if st.checkbox('View results as table'):
                    st.write(model_leaderboard[['model','score_val']])

#### Plot model output
                if st.checkbox('View plots'): 
                    fig,ax = plt.subplots()
                    ax.barh(model_leaderboard['model'],model_leaderboard['score_val'])
                    plt.xlabel('Validation score')
                    plt.ylabel('Model')
                    #plt.xlim(1)
                    st.write(fig)
                
                    trained_models = model_leaderboard['model']
