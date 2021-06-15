
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

## Import modules from python scripts
from Preprocessing_Modules import get_predictive_power_score,drop_variables,feature_generator,aggregated_data
from Model_Training_Modules import manual_model_training_classification,manual_model_training_regression,func_train_test_split,func_classification,func_regression


st.set_page_config(layout='wide')

st.header('                                                    Open source Machine Learning app')

st.sidebar.markdown('This app helps to train classification and regression(coming soon) in manual and automated mode. For Manual classification tasks sklearn LogisticRegression and RandomForest are used in default settings. For automated model training [Autogluon](https://auto.gluon.ai/stable/index.html) library is used.')

with st.sidebar.beta_expander('Features'):
    
    st.markdown('#### View dataset details: ')
    st.markdown('- Use sample datasets (Iris, Boston) ')
    st.markdown('- Upload CSV file ')
    st.markdown('#### View dataset details: ')
    st.markdown('- Row and Column counts')
    st.markdown('- View sample data rows upto 20')
    st.markdown('#### Data type conversion: ')
    st.markdown('- To Object')
    st.markdown('- To Numeric')
    st.markdown('- To Datetime')
    st.markdown('#### Data visualization: ')
    st.markdown('- Univariate')
    st.markdown('- Bivariate (Numeric only)')
    st.markdown('#### Feature Generation: ')
    st.markdown('- Generate aggregated features based on a grouping column')
    st.markdown('#### Data Processing: ')
    st.markdown('- Drop / Impute (Mean, Mode) Nulls')
    st.markdown('- Remove high frequency columns')
    st.markdown('- Encode Data using getdummies()')
    st.markdown('#### Predictive Power: ')
    st.markdown('- Find highest predictive variables using ppscore')
    st.markdown('#### Model Training: ')
    st.markdown(' - Manual (Classification / Regression) using Linear regression, Logistic regression and RandomForest')
    st.markdown('- Automated, using Autogluon')

with st.sidebar.beta_expander('Upcoming'):
    st.markdown('- LIME prediction explanation')

st.subheader('Select options: ')

with st.beta_expander('Data selection'):

    st.markdown('Select option to upload your own dataset in csv format or select from existing sample datasets.')
    st.markdown('Iris dataset is used for Classification and Boston dataset is used for Regression tasks.')

    option = st.selectbox('',['Sample datasets','Upload CSV'])

    if option == 'Sample datasets':

        st.markdown('Select problem: ')
        option_prob = st.selectbox('',['Regression','Classification'])

        if option_prob == 'Regression':
            from sklearn.datasets import load_boston
            dataset = 'temp'
            dataset_x = load_boston()
            input_dataset = pd.DataFrame(dataset_x.data,columns= dataset_x.feature_names)
            input_dataset['target'] = dataset_x.target
            input_dataset.columns = input_dataset.columns.str.lower()
        if option_prob == 'Classification':
            from sklearn.datasets import load_iris
            dataset = 'temp'
            dataset_x = load_iris()
            input_dataset = pd.DataFrame(dataset_x.data,columns= dataset_x.feature_names)
            input_dataset['target'] = dataset_x.target
            input_dataset.columns = input_dataset.columns.str.lower()

    if option == 'Upload CSV':

        dataset = st.file_uploader("",)

       
        if dataset is not None:

            try:

#### Read dataset from upload
                input_dataset = pd.read_csv(dataset)

#### Change column names to lower case
                st.success('File upload successful')

            except ValueError:
                st.error('Something went wrong while uploading the dataset')
        else:
            st.markdown('### Dataset not found. Please upload a dataset first in CSV format')
    

st.markdown('------------------------------------------------------------------------------------------------------------')
    
#### Dataset details section    
try:
            
    with st.beta_expander('View dataset details'):
            
        st.markdown('#### Number of records: ')
        st.write(input_dataset.shape[0])
            
        st.markdown('#### Number of columns: ')
        st.write(input_dataset.shape[1])

        if st.checkbox('Show sample'):

            record_count = st.slider('Select number of records to show',1,20)
                
            st.write(input_dataset.head(record_count))
    
    st.markdown('------------------------------------------------------------------------------------------------------------')

    with st.beta_expander('Data type conversion'):

        dataset_integer_variables = input_dataset.select_dtypes(['int','float']).columns
        dataset_object_variables = input_dataset.select_dtypes('O').columns
        dataset_date_variables = input_dataset.select_dtypes(['datetime']).columns
        
        selected_columns = st.multiselect('Select features for datatype conversion',input_dataset.columns)

        datatype_selection = st.selectbox('Select datatype to convert to',['','Integer','Object','Datetime'])
        
        feature_converted_dataset = input_dataset

        if datatype_selection=="Datetime":
            
            try:
                
                for i in selected_columns:
                    
                    if i not in dataset_date_variables:
                       
                        feature_converted_dataset[i] = pd.to_datetime(feature_converted_dataset[i])
                
                    st.success('Data type conversion complete')


            except ValueError:
                
                st.error('Seems there is a problem with your variable selection')
    
        if datatype_selection=="Integer":
            
            try:
                
                for i in selected_columns:
                    
                    if i not in dataset_integer_variables:
                       
                        feature_converted_dataset[i] = feature_converted_dataset[i].astype('int')
                
                    st.success('Data type conversion complete')

            except ValueError:
                st.error('Seems there is a problem with your variable selection')
    
        if datatype_selection =="Object":
            
            try:
            
                for i in selected_columns:
                    
                    if i not in dataset_object_variables:
                        
                        feature_converted_dataset[i] = feature_converted_dataset[i].astype('O')
                    
                    st.success('Data type conversion complete')
            
            except ValueError:
                st.error('Seems there is a problem with your variable selection')


    st.markdown('------------------------------------------------------------------------------------------------------------')

except:
    st.error('Something is wrong here, check your input')
#### Data visualization section
try:
    with st.beta_expander('Data Visualization'):

        st.markdown('#### Select visualization type')
        visual_type = st.radio('',('Univariate','Bi-variate'))

        if feature_converted_dataset is not None:

            visualization_dataset = feature_converted_dataset
        
        else:

            visualization_dataset = input_dataset

        if visual_type == 'Univariate':
                
            st.markdown('#### Select variables to plot')
            variables = st.selectbox('',visualization_dataset.columns)

            st.markdown('#### Adjust Figure width')
            fig_w = st.slider('',5,20,key='st_1')
                
            st.markdown('#### Adjust Figure height')
            fig_h = st.slider('',2,20,key='st_2')

            if variables in dataset_integer_variables:
                
                fig,ax = plt.subplots(figsize =(fig_w,fig_h))
                ax.hist(visualization_dataset[variables])
                st.pyplot(fig)

            if variables in dataset_object_variables:
                    
                height = st.selectbox('select x-axis',dataset_integer_variables)
                fig,ax = plt.subplots(figsize =(fig_w,fig_h))
                ax.bar(visualization_dataset[variables],visualization_dataset[height])
                st.pyplot(fig)
        
        if visual_type == "Bi-variate":
                
            st.write('Only numeric variables supported. Updates coming soon')

            x = st.selectbox('Select variables to plot on x axis',dataset_integer_variables)
                
            y = st.selectbox('Select variables to plot on y axis',dataset_integer_variables)

            fig_w_b = st.slider('Adjust Figure width',5,20,key='st_5')
            fig_h_b = st.slider('Adjust Figure height',2,20,key='st_6')
        
            fig,ax = plt.subplots(figsize = (fig_w_b,fig_h_b))
            ax.scatter(visualization_dataset[x],visualization_dataset[y])
            plt.xlabel(x)
            plt.ylabel(y)
            st.pyplot(fig)
    st.markdown('------------------------------------------------------------------------------------------------------------')
except:
    st.error('Somethin is wrong here, check your input')
    

try:
    with st.beta_expander('Feature Generation'):

        st.write('The preprocessing methods to be updated for better flow')

        col6,col7 = st.beta_columns(2)

        with col6:
            operations = st.multiselect('',['Remove High frequency columns','Encode categorical columns','Process Nulls'])
        
        with col7:
            for i in operations:
                #try:
                if i == 'Remove High frequency columns':
                    
                    high_frequency = drop_variables(input_dataset)

                    input_dataset = input_dataset.drop(high_frequency,axis = 1)
                    
                    #dataset_integer_variables = input_dataset.select_dtypes(include = ['int','int32','float','float32']).columns
                    #dataset_object_variables = input_dataset.select_dtypes(include = ['O']).columns
                
                if i == 'Encode categorical columns':

                    input_dataset = pd.get_dummies(input_dataset)               

                if i == 'Process Nulls':
                        
                    Null_columns = input_dataset.isnull().mean()

                    Null_process_method = st.multiselect('Select null imputation method: ',['Fill NA','Drop'])     

                    for j in Null_process_method:
                            
                        if j=="Drop":

                            input_dataset = input_dataset.dropna(thresh = 0.8,axis = 1)
                        
                        if j=="Fill NA":
                            
                            for k in dataset_integer_variables:
                                input_dataset[k] = input_dataset[k].fillna(np.mean(input_dataset[k]))

                            for k in dataset_object_variables:
                                input_dataset[k] = input_dataset[k].fillna(input_dataset[k].mode())     
        
        
        st.markdown('#### Generate date features: ')

        date_feature_generator = st.multiselect('',[None,'Year','Month','Quarter','Week of Year','Day of Week','Weekday Name'])

        if date_feature_generator is not None:

            input_dataset = feature_generator(input_dataset,date_feature_generator)

        st.write(input_dataset.head(2))
                #except:
                 #  st.error('Something went wrong during data preprocessing')
        
        st.markdown('#### Generate Aggregations')

        st.markdown('\n ##### Select Grouping column \n')
        group_col = st.selectbox('',input_dataset.select_dtypes(include = ['O']).columns,key = 'g_1'  )

        st.markdown('\n##### Select column to aggregate')
        agg_col = st.selectbox('',input_dataset.select_dtypes(include = ['int','int64','float','float64']).columns,key = 'a_1')

        st.markdown('##### Select Aggregation')
        aggregation = st.multiselect('',['Sum','Average'])

        working_dataset = aggregated_data(group_col,agg_col,aggregation,input_dataset)

        st.write(working_dataset.head())
        st.write('Updates Coming soon')
    st.markdown('------------------------------------------------------------------------------------------------------------')
except:
    st.error('Somethin is wrong here, check your input')
    

try:
    with st.beta_expander('Predictive Power'):

        target_var = st.selectbox('Select Target Variable',working_dataset.columns)

        predictive_p,high_predictive_variables = get_predictive_power_score(working_dataset,target_var)

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

            #st.write(updated_dataset_with_selected_features)
    st.markdown('------------------------------------------------------------------------------------------------------------')
except:
    st.error('Somethin is wrong here, check your input')    
    

try:
    with st.beta_expander('Model Training'):

        if selected_features==True:

            working_dataset = updated_dataset_with_selected_features
        
        else:

            working_dataset = input_dataset

#        modeling_dataset = working_dataset
        
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

                    target = st.selectbox('Select Target variable (Integer only)',working_dataset.columns)

                    st.write('\n Training set  size (%) : ',training_size)
                    
                    st.write('\n Test set size (%) :',test_size)

                    train_X,test_X,train_y,test_y = train_test_split(working_dataset.drop(target,axis = 1),working_dataset[target],test_size=test_size,random_state = 42)
                
                if st.checkbox('Start Training'):

                    predictor,score = manual_model_training_regression(model_type,train_X,train_y,test_X,test_y)

                    if model_type == 'Linear Regression':

                        st.write('\n Linear Regression R2 Value: ',score)
                
                    if model_type == 'Random Forest':

                        st.write('\n Random Forest MAE Value: ',score)

            if problem_type =='Classification':

                with col3:
                
                    model_type = st.radio('Select training algorithm',['Logistic Regression','Random Forest'])
                    
                    training_size = st.slider('Select Training set size',0.1,1.0,step=0.1,)
                    
                    
                with col4:
                    
                    target = st.selectbox('Select Target variable',working_dataset.columns)

                    target = target.lower()
                    
                    test_size = np.round(1-training_size,2)

                    if st.checkbox('\n Convert target to object type?'):
                    
                        working_dataset[target] = working_dataset[target].astype('O')

                    st.write('\n Training set  size (%) : ',training_size)
                    
                    st.write('\n Test set size (%) :',test_size)

                    train_X,test_X,train_y,test_y = train_test_split(working_dataset.drop(target,axis = 1),working_dataset[target],test_size=test_size,random_state = 42)
                    
                if st.checkbox('Start Training'):

                    accuracy,f1_score,confusion_matrix,predictor = manual_model_training_classification(model_type,train_X,train_y,test_X,test_y)

                    st.write(model_type,' Accuracy: ',accuracy)
                    st.write(model_type,' F1 Score: ',f1_score)
                    st.table(confusion_matrix)
                        
                    if st.checkbox('Export Model'):
                        
                        import pickle

                        pickle.dump(predictor,open(model_type+'_export.sav','wb'))


        if training_method=='Automated Model Training':

            st.markdown('This method uses [Autogluon](https://auto.gluon.ai/stable/index.html) library for automated model training and comparison')
    
            col5,col6 = st.beta_columns(2)
            
            with col5:

                problem_type = st.radio('Select problem type',['Regression','Classification'])
                target_variable = st.selectbox("Select target variable",working_dataset.columns)
                target_variable = target_variable.lower()
            
                if st.checkbox('Convert target to object type?'):

                    modeling_dataset[target_variable] = working_dataset[target_variable].astype('O')          
            
            with col6:

                size_train = np.round(st.slider('Training size',0.1,1.0),2)
                size_test = np.round(1-(size_train),2)
                st.write('Training set size: ',size_train)
                st.write('\nTest set size: ',size_test)

            if size_train>0:

                test_data = working_dataset.sample(frac = size_test)
                train_data = working_dataset.sample(frac = size_train)
            
                if st.checkbox('Start model analysis'):
                    progress_bar = st.progress(0)

                    for compl in range(100):
                        time.sleep(1)
                        progress_bar.progress(compl+1)

                    if problem_type == 'Classification':

                        predictor,model_leaderboard = func_classification(train_data,test_data,target_variable)
                    
                    if problem_type == 'Regression':
                        
                        predictor,model_leaderboard = func_regression(train_data,test_data,target_variable)
                    
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
    st.markdown('------------------------------------------------------------------------------------------------------------')

except:
    st.error('Somethin is wrong here, check your input')
    
#try:
with st.beta_expander('Test Predictions (Currently available for sample datasets only)'):

    if option == 'Sample datasets':
        if working_dataset is not None :

            working_dataset_pred = working_dataset.drop(target,axis = 1)

            categorical_columns = working_dataset_pred.select_dtypes(include = ['O']).columns
            numeric_columns = working_dataset_pred.select_dtypes(include = ['int','float']).columns

            col_7,col_8 = st.beta_columns(2)

            with col_7:

                st.markdown('## Select values')
                pred_list = []
                col_name = []

                for i in categorical_columns:
                    st.write(i)
                    selection = st.selectbox(i,working_dataset_pred[i].unique())

                for i in numeric_columns:

                    max = working_dataset_pred[i].max()
                    st.markdown(i.upper())
                    selection_numeric = st.slider('',min_value = 0.0,max_value = float(max),key = i)
                    col_name.append(i)
                    pred_list.append(selection_numeric)

            df = pd.DataFrame([],columns = col_name)

            df = df.append(pd.Series(pred_list,index = col_name),ignore_index = True)

            with col_8:
                
                if problem_type =='Regression':

                    import lime
                    import lime.lime_tabular
                
                    df_pred = df
                    predicted = predictor.predict(df_pred)
                
                    predicted_for_lime = lambda x: predictor.predict(x.to_numpy())
                
                
                    st.write('### Predicted Outcome: ',*np.round(predicted,2))
                
                    lime_explainer = lime.lime_tabular.LimeTabularExplainer(train_X.values,feature_names = train_X.columns,mode = 'regression')

                    explanation_output = lime_explainer.explain_instance(df_pred.iloc[0],predictor.predict)

                    st.markdown('### What caused this output: ')
                    st.write(explanation_output.as_pyplot_figure())
                
                if problem_type =='Classification':


                    df_pred = df
                    predicted = predictor.predict(df_pred)
                
                    predicted_for_lime = lambda x: predictor.predict(x.to_numpy())
                
                
                    st.write('### Predicted Outcome: ',*np.round(predicted,2))
                
                    lime_explainer = lime.lime_tabular.LimeTabularExplainer(train_X.values,feature_names = train_X.columns)

                    explanation_output = lime_explainer.explain_instance(df_pred.iloc[0],predictor.predict)

                    st.markdown('### What caused this output: ')
                    st.write(explanation_output.as_pyplot_figure())
    else:
        st.markdown('#### Feature not available')
                
