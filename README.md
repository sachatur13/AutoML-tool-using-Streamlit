# Modeling tool using Streamlit / Sklearn / Pandas / Autogluon

Creating a simple model analysis tool using 
- Streamlit framework as UI
- Scikit-learn for manual machine learning models

#### Current options - 
- View dataset details
    - Row and Column counts
    - View sample data rows upto 20
- Data type conversion
    - To Object
    - To Numeric
    - To Datetime
- Data visualization
    - Univariate
    - Bivariate (Numeric only)
- Feature Generation:
    - Generate aggregated features based on a grouping column
- Data Processing
    - Drop / Impute (Mean, Mode) Nulls
    - Remove high frequency columns
    - Encode Data using getdummies()
- Predictive Power: 
    - Find highest predictive variables using ppscore
- Model Training:
    - Manual (Classification / Regression) using Linear regression, Logistic regression and RandomForest
    - Automated, using Autogluon

##### Classification

        Logistic Regression
        Random Forest

##### Regression

        Linear Regression
        Random Forest
        
The models are trained in default mode without any hyperparameter tuning.

- Data visualization using matplotlib.
- Data preprocessing using pandas.
- Autogluon for Automated model training and leaderboard.
- Model export as pickle file for manual trained models using sklearn.
- Predictive power score for feature importance using ppscore library.

To be added features - 
Model deployments
S3 connection
