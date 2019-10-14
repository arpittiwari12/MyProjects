import sys
sys.path.append('..')

from data import make_dataset as md
from preprocessing import preprocess as pp
import time

from sklearn import cross_validation, metrics

OUTCOME_NAME = "FLAG"

def sklearn_from_pandas(df, outcome = OUTCOME_NAME):
    '''Turns dataframe to sklearn api objects
    
    Parameters
    ----------
    df : pandas.DataFrame
        Cleaned, entirely float dataframe
    
    Returns
    -------
    X : numpy.ndarray
        model features as numpy matrix
    y : numpy.ndarray
        1-dimensional array of outcome labels
    feature_names : list[str]
        list of feature names from pandas columns
    '''
    feature_names = df.columns.difference([outcome])
    
    X = df[feature_names].as_matrix()
    y = df[outcome].values
    
    return(X, y, feature_names)

def performance_metrics(X_test, y_test, trained_model):
    '''Dict of ML model kpis
    
    Parameters
    ----------
    X_test : numpy.ndarray
        reserved features to test prediction
    y_test : numpy.ndarray
        reserved labels to test prediction
    trained_model : sklearn.base.ClassifierMixin
        sklearn classifying estimator, pre-trained
    
    Returns
    -------
    ml_metrics : dict
        dictionary with model score, F1, and auc
    '''
    y_pred = trained_model.predict(X_test)
    y_score = trained_model.predict_proba(X_test)[:, 1]
    
    ml_metrics = dict()
    ml_metrics['accuracy'] = metrics.accuracy_score(y_test, y_pred)
    ml_metrics['f1'] = metrics.f1_score(y_test, y_pred)
    ml_metrics['auc'] = metrics.roc_auc_score(y_test, y_score)
    
    return(ml_metrics)

def parse_data(df, printing=False):
    '''Parses RP data for sklearn
    
    Parameters
    ----------
    query : str
        PL SQL Query
    printing : bool, default False
        print benchmarking output
        
    Returns
    -------
    pandas.DataFrame
        all numeric df, prepared for sklearn
    '''
    start_time = time.time()
    
    # read data from AZCFL
    #df_raw = md.load_data_from_query(query)
    df_raw = df
    if printing:
        print("== DATA PARSING ==")
        print("Data loaded from EDW")
        curr_time = time.time() - start_time
        print("Time: {0} s, Shape: {1}".format(int(curr_time), df_raw.shape))
    
    # drop columns with >10% nulls, impute 0s
    df = md.drop_null_columns(df_raw)
    df = md.impute_zeros(df)
    if printing:
        print("")
        print("Columns with nulls -- null count")
        print(md.columns_nulls(df))
    
    # impute outcome flag
    pp.impute_flag(df, outcome = OUTCOME_NAME)
    if printing:
        print("")
        print("{0} Count by AOV Band".format(OUTCOME_NAME))
        print(df.groupby(['AOV_BAND', 'FLAG']).size())
        
    # encode factor columns
    #pp.order_feature(df, 'ACCOUNT_TYPE')
    model_data, variables = pp.subset_encode_df_jorge(df, outcome = OUTCOME_NAME)  # Change it with mine
    if printing:
        print("")
        print("== DONE DATA PARSING ==")
        final_time = time.time() - start_time
        print("Final Time: {0} s, Final Shape: {1}".format(int(final_time), model_data.shape))
    
    
    return(model_data, variables)

def basic_train_model(model_data, model, outcome=OUTCOME_NAME, printing=False):
    '''Trains a model and returns metrics
    
    Parameters
    ----------
    model_data : pandas.DataFrame
        Cleaned, entirely float dataframe
    model : sklearn.base.ClassifierMixin
        untrained sklearn classfier
    outcome : str
        outcome column name in dataframe
    printing : boolean, default False
        print benchmarking output
        
    Returns
    -------
    ml_metrics : dict
        dictionary with model score, F1, and auc
    '''
    X, y, feature_names = sklearn_from_pandas(model_data, outcome)
    
    X_train, X_test, y_train, y_test = \
        cross_validation.train_test_split(X,
                                          y,
                                          test_size=0.3,
                                          random_state=123)
    if printing:
        print('Training: {0} rows'.format(len(y_train)))
        print('Testing: {0} rows'.format(len(y_test)))
        start_time = time.time()
        
    model.fit(X_train, y_train)
    
    if printing:
        fit_time = time.time() - start_time
        print("Fit in {0:.2f} s".format(fit_time))
        
    ml_metrics = performance_metrics(X_test, y_test, model)
    
    if printing:
        print('Accuracy: {0:.4f}'.format(ml_metrics['accuracy']))
        print('F1:       {0:.4f}'.format(ml_metrics['f1']))
        print('AUC:       {0:.4f}'.format(ml_metrics['auc']))
    
    return(ml_metrics)