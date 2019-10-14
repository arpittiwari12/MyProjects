import numpy as np
from numpy import linspace
from matplotlib import pyplot

#To be implemented:
#def plot_lift_curve():
#def plot_variables_hist():
#def plot_cols_cov():

def filter_data(data, col_name='AOV', filter_name='aov:10_50'):
    if filter_name == 'aov:1_10':
        return data[(data[col_name] <= 10000) & (data[col_name] > 1000)]
    elif filter_name == 'aov:10_50':
        return data[(data[col_name] <= 50000) & (data[col_name] > 10000)]
    elif filter_name =='aov:50_200':
        return data[(data[col_name] <= 200000) & (data[col_name] > 50000)]
    elif filter_name =='aov:200_600':
        return data[(data[col_name] <= 600000) & (data[col_name] > 200000)]
    elif filter_name =='aov:100_200':
        return data[(data[col_name] <= 200000) & (data[col_name] > 100000)]
    elif filter_name =='aov:100_600':
        return data[(data[col_name] <= 600000) & (data[col_name] > 100000)]
    elif filter_name =='aov:50_100':
        return data[(data[col_name] <= 100000) & (data[col_name] > 50000)]
    else: 
        return data
    
def get_risk_segmentation_split(data, aov_col='AOV', age_col='ACCOUNT_AGE', segment='>180_days_aov10_50'):
    if segment == '>180_days_aov1_10':
        return data[(data[aov_col] <= 10000) & (data[aov_col] > 1000) & (data[age_col] > 180)]
    elif segment == '>180_days_aov10_50':
        return data[(data[aov_col] <= 50000) & (data[aov_col] > 10000) & (data[age_col] > 180)]
    elif segment == '>180_days_aov50_100':
        return data[(data[aov_col] <= 100000) & (data[aov_col] > 50000) & (data[age_col] > 180)]
    elif segment == '>180_days_aov100_600':
        return data[(data[aov_col] <= 600000) & (data[aov_col] > 100000) & (data[age_col] > 180)]
    elif segment == '<=180_days_aov1_10':
        return data[(data[aov_col] <= 10000) & (data[aov_col] > 1000) & (data[age_col] <= 180)]
    elif segment == '<=180_days_aov10_50':
        return data[(data[aov_col] <= 50000) & (data[aov_col] > 10000) & (data[age_col] <= 180)]
    elif segment == '<=180_days_aov50_100':
        return data[(data[aov_col] <= 100000) & (data[aov_col] > 50000) & (data[age_col] <= 180)]
    elif segment == '<=180_days_aov100_600':
        return data[(data[aov_col] <= 600000) & (data[aov_col] > 100000) & (data[age_col] <= 180)]
    
def top_corr(data, n=10, remove_ones=True):
    # This version removes correlations of 1 to remove self-correlations. This misses the variables with correlations = 1
    corr = data.corr()
    orde = corr[~corr.isnull().all(axis=1)].dropna(axis=1).unstack().order(kind="quicksort")
    if remove_ones:
        orde = orde[orde != 1]
    return orde[-n:]

def top_corr_neg(data, n=10):
    corr = data.corr()
    orde = corr[~corr.isnull().all(axis=1)].dropna(axis=1).unstack().order(kind="quicksort")
    return orde[:n]
    
def split_to_x_y(data, y_column, x_exclusion_list= None):
    X = data.drop([y_column], axis=1)
    if x_exclusion_list is not None:
        X = data.drop(x_exclusion_list, axis=1)
    Y = data[y_column]
    return X, Y

# Randomizing/ shuffling
def randomize(dataset, labels):
  permutation = np.random.permutation(labels.shape[0])
  shuffled_dataset = dataset[permutation,:]
  shuffled_labels = labels[permutation]
  return shuffled_dataset, shuffled_labels


def run_modeling(model, train_data, train_label, test_data, test_label, parameter=None, p_range=None, lc=True, cvc=True, scoring='f1_weighted'):
    
    from sklearn import cross_validation
    from sklearn.metrics import roc_curve, auc
    
    model.fit(train_data, train_label)
    print ("Training completed for model: %s" %model.__class__)
    print ("Model parameters: %s" %model.get_params())

    score = model.score(test_data, test_label)
    scores = cross_validation.cross_val_score(model, train_data, train_label, cv=6, scoring=scoring)
    
    pred_y_prob = model.predict_proba(test_data)[:,1]
    pred_y = model.predict(test_data)
    fpr, tpr, threshold = roc_curve(test_label, pred_y_prob)
    auc = auc(fpr, tpr)

    # Creating plots
    plot_perf_plots(test_label, pred_y, y_prob=pred_y_prob, title="model performance")
    if lc:
        plot_learning_curve(model, train_data, train_label, train_sizes=np.linspace(.2, 1, 10), scoring=scoring)
    if cvc:
        if parameter is None or p_range is None:
            raise AttributeError('Missing CVC attributes')
        else:
            plot_cv_curve(model,train_data,train_label, parameter, p_range, scoring=scoring)

    #Will also update data_test to include probability
    #plot_aov_perf_plots(data_test, pred_y_prob,'ATT_AOV', 'NON_ATT_AOV')
    #plot_aov_perf_plots(data_test, pred_y_prob,'ATT_FLAG', 'NON_ATT_FLAG')
    
    return score, scores, auc

def run_evaluation(model, test_data, test_label, parameter=None, p_range=None, lc=True, cvc=True, scoring='f1_weighted'):
    
    from sklearn import cross_validation
    from sklearn.metrics import roc_curve, auc
    
    print ("Model parameters: %s" %model.get_params())

    score = model.score(test_data, test_label)
    scores = cross_validation.cross_val_score(model, test_data, test_label, cv=6, scoring=scoring)
    
    pred_y_prob = model.predict_proba(test_data)[:,1]
    pred_y = model.predict(test_data)
    fpr, tpr, threshold = roc_curve(test_label, pred_y_prob)
    auc = auc(fpr, tpr)

    # Creating plots
    plot_perf_plots(test_label, pred_y, y_prob=pred_y_prob, title="model performance")
    if lc:
        plot_learning_curve(model, test_data, test_data, train_sizes=np.linspace(.2, 1, 10), scoring=scoring)
    if cvc:
        if parameter is None or p_range is None:
            raise AttributeError('Missing CVC attributes')
        else:
            plot_cv_curve(model,test_data,test_data, parameter, p_range, scoring=scoring)

    #Will also update data_test to include probability
    #plot_aov_perf_plots(data_test, pred_y_prob,'ATT_AOV', 'NON_ATT_AOV')
    #plot_aov_perf_plots(data_test, pred_y_prob,'ATT_FLAG', 'NON_ATT_FLAG')
    
    return score, scores, auc
    
def fit_model(model, train_data, train_label):
    
    from sklearn import cross_validation
    from sklearn.metrics import roc_curve, auc
    
    model.fit(train_data, train_label)
    print ("Training completed for model: %s" %model.__class__)
    print ("Model parameters: %s" %model.get_params())

    score = model.score(train_data, train_label)
    #scores = cross_validation.cross_val_score(model, train_data, train_label, cv=6, scoring=scoring)
    
    #pred_y_prob = model.predict_proba(test_data)[:,1]
    #pred_y = model.predict(test_data)
    #fpr, tpr, threshold = roc_curve(test_label, pred_y_prob)
    #auc = auc(fpr, tpr)
    
    return score

def run_experiment (model, output_column, test_size = .3, dataset=None, reload_data= False, drop_na=True, scaler=None, dim_red=None, feature_extractor=None, output_scaler=None, output_scaler_param=None, filter_name=None, filter_col_name='AOV', projected_drop_columns=None, projected_select_columns=None, input_exclusion_columns_list=None, data=None, task='NA', sql_file=None, query_name=None, save_experiment=False, experiment_file=None, subset_size=1000, use_all_dataset=False, lc=False, cvc=False, cv_param=None, cv_p_range=None, scoring='f1', remove_processed_features=False):
    
    
    from sklearn import cross_validation
    import connect_warehouse, experiments, encoding
    import pandas as pd
    
    if dataset is None or reload_data:
        # Pulling data from DW (You need to write a sql file with the queries)
        print("Pulling data from Warehouse")
        res = connect_warehouse.get_data_using_sql_file(query_name, sql_file)
    else:
        print("Using existing dataset")
        res = dataset
    
    subset = res.copy()
    print("Raw data: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    #Filtering unrelevant datapoints
    if filter_name is not None:
        subset = filter_data(subset, col_name=filter_col_name, filter_name=filter_name)
        print("Filtered: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    # Subsetting for faster computations (if working on prototyping)
    if not use_all_dataset:
        #subset = subset[:subset_size]
        subset = subset.sample(subset_size)
        print("Subsetted: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    
    if output_scaler is not None:
        if output_scaler_param is not None:
            subset['LABEL'] = subset[output_column].apply(lambda x: output_scaler(x, output_scaler_param))
        else:
            subset['LABEL'] = subset[output_column].apply(lambda x: output_scaler(x))
    else:
        subset['LABEL'] = subset[output_column]
    
    print("After Creating Label column: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    
    # Projecttion by selection
    if projected_select_columns is not None:
        select_list = projected_select_columns.copy()
        select_list.append('LABEL')
        #projected_select_columns.append(output_column)
        subset = subset[select_list]
        print("Projected after columns selection: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    
    # Feature extraction, adding calculated columns, pivoting
    if feature_extractor is not None:
        subset, converted_columns = feature_extractor(subset)
        print("After adding features: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
        
    # Projection by elimination - needs to be removed
    if projected_drop_columns is not None:
        subset = subset.drop(projected_drop_columns, axis=1)
        print("Projected after columns removal: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    # Removing original features
    if remove_processed_features and converted_columns is not None:
        subset = subset.drop(converted_columns, axis=1)
        print("Projected after original features removal: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
        print(subset.count())
    
    # Check & removing missing data
    print("Number of datapoints with NaN: %s" %subset[subset.isnull().any(axis=1)].shape[0])

    # drop NaNs
    if drop_na:
        subset = subset.dropna()
        print("After dropping NaNs: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    # Splitting to input and output
    X, Y = split_to_x_y(subset, 'LABEL', input_exclusion_columns_list)
    print("X dimension: %s rows and %s columns" %(X.shape[0], X.shape[1]))
    
    trained_scalaer, trained_dim_red = None, None
    #Scalilng
    if scaler is not None:
        vector = X.as_matrix()
        scaler = scaler.fit(vector)
        vector_scaled = scaler.transform(vector)
        print("After Scaling Input: %s rows and %s columns" %(vector_scaled.shape[0], vector_scaled.shape[1]))

        #to dataframe
        X = pd.DataFrame(vector_scaled)
    
    #Dimensionality reduction
    if dim_red is not None:
        print("X original dimension is: (%s)" %X.shape[1])
        from sklearn.feature_selection import SelectFromModel
        dim_red = dim_red.fit(X)
        #dim_model = SelectFromModel(reshaper, prefit=True)
        X = dim_red.transform(X)
        print("X new dimension after dimensionality reduction is: (%s)" %X.shape[1])
    

    # Splitting to train/test
    train_x, test_x, train_y, test_y = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=0)
    print("Training set: %s rows and %s columns" %(train_x.shape[0], train_x.shape[1]))
    print("Testing set: %s rows and %s columns" %(test_x.shape[0], test_x.shape[1]))

    # Shuffling data
    #train_dataset, train_labels = randomize(train_x, train_y)
    
    # Building models
    score, scores, auc = run_modeling(model, train_data=train_x, train_label=train_y, test_data=test_x, test_label=test_y ,lc=lc, cvc=cvc, parameter=cv_param, p_range=cv_p_range, scoring=scoring)
    
    # Saving experiment
    if experiment_file is not None and save_experiment:
        exper = experiments.Experiments(experiment_file)
        exp = exper.template

        exp['Task'] = task
        exp['Data'] = "Query name: %s Subset: %s Filter: %s" %(query_name, subset_size, filter_name) 
        exp['Features'] = "Excluded columns: %s" %projected_drop_columns
        exp['Preprocessing'] = "Scaling: %s Dimensionality: %s" %(scaler, dim_red)
        exp['Details'] = "Test Size: %s" %test_size
        exp['Algorithm'] = model.__class__
        exp['Parameters'] = model.get_params()

        # Saving results
        exp['Evaluation'] = "CV F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
        exp['F1'] = score
        exp['AUC'] = auc
        exper.save(exp)
        print("Experiment saved")
        
    return res, model, scaler, dim_red
   
    
def run_fitting (model, output_column, dataset=None, reload_data= False, drop_na=True, scaler=None, dim_red=None, feature_extractor=None, output_scaler=None, output_scaler_param=None, filter_name=None, filter_col_name='AOV', projected_drop_columns=None, projected_select_columns=None, input_exclusion_columns_list=None, data=None, task='NA', sql_file=None, query_name=None, save_experiment=False, experiment_file=None, subset_size=1000, use_all_dataset=False, cv_param=None, cv_p_range=None, scoring='f1', remove_processed_features=False):
    
    
    from sklearn import cross_validation
    import connect_warehouse, experiments, encoding
    import pandas as pd
    
    if dataset is None or reload_data:
        # Pulling data from DW (You need to write a sql file with the queries)
        print("Pulling data from Warehouse")
        res = connect_warehouse.get_data_using_sql_file(query_name, sql_file)
    else:
        print("Using existing dataset")
        res = dataset
    
    subset = res.copy()
    print("Raw data: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    #Filtering unrelevant datapoints
    if filter_name is not None:
        subset = filter_data(subset, col_name=filter_col_name, filter_name=filter_name)
        print("Filtered: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    # Subsetting for faster computations (if working on prototyping)
    if not use_all_dataset:
        #subset = subset[:subset_size]
        subset = subset.sample(subset_size)
        print("Subsetted: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    
    if output_scaler is not None:
        if output_scaler_param is not None:
            subset['LABEL'] = subset[output_column].apply(lambda x: output_scaler(x, output_scaler_param))
        else:
            subset['LABEL'] = subset[output_column].apply(lambda x: output_scaler(x))
    else:
        subset['LABEL'] = subset[output_column]
    
    print("After Creating Label column: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    
    # Projecttion by selection
    if projected_select_columns is not None:
        select_list = projected_select_columns.copy()
        select_list.append('LABEL')
        #projected_select_columns.append(output_column)
        subset = subset[select_list]
        print("Projected after columns selection: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    
    # Feature extraction, adding calculated columns, pivoting
    if feature_extractor is not None:
        subset, converted_columns = feature_extractor(subset)
        print("After adding features: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
        
    # Projection by elimination - needs to be removed
    if projected_drop_columns is not None:
        subset = subset.drop(projected_drop_columns, axis=1)
        print("Projected after columns removal: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    # Removing original features
    if remove_processed_features and converted_columns is not None:
        subset = subset.drop(converted_columns, axis=1)
        print("Projected after original features removal: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
        print(subset.count())
    
    # Check & removing missing data
    print("Number of datapoints with NaN: %s" %subset[subset.isnull().any(axis=1)].shape[0])

    # drop NaNs
    if drop_na:
        subset = subset.dropna()
        print("After dropping NaNs: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    # Splitting to input and output
    X, Y = split_to_x_y(subset, 'LABEL', input_exclusion_columns_list)
    print("X dimension: %s rows and %s columns" %(X.shape[0], X.shape[1]))
    
    trained_scalaer, trained_dim_red = None, None
    #Scalilng
    if scaler is not None:
        vector = X.as_matrix()
        scaler = scaler.fit(vector)
        vector_scaled = scaler.transform(vector)
        print("After Scaling Input: %s rows and %s columns" %(vector_scaled.shape[0], vector_scaled.shape[1]))

        #to dataframe
        X = pd.DataFrame(vector_scaled)
    
    #Dimensionality reduction
    if dim_red is not None:
        print("X original dimension is: (%s)" %X.shape[1])
        from sklearn.feature_selection import SelectFromModel
        dim_red = dim_red.fit(X)
        #dim_model = SelectFromModel(reshaper, prefit=True)
        X = dim_red.transform(X)
        print("X new dimension after dimensionality reduction is: (%s)" %X.shape[1])
    

    # Splitting to train/test
    #train_x, test_x, train_y, test_y = cross_validation.train_test_split(X, Y, test_size=test_size, random_state=0)
    #print("Training set: %s rows and %s columns" %(train_x.shape[0], train_x.shape[1]))
    #print("Testing set: %s rows and %s columns" %(test_x.shape[0], test_x.shape[1]))

    # Shuffling data
    #train_dataset, train_labels = randomize(train_x, train_y)
    
    # Building models
    score = fit_model(model, train_data=X, train_label=Y)
    
    # Saving experiment
    if experiment_file is not None and save_experiment:
        exper = experiments.Experiments(experiment_file)
        exp = exper.template

        exp['Task'] = task
        exp['Data'] = "Query name: %s Subset: %s Filter: %s" %(query_name, subset_size, filter_name) 
        exp['Features'] = "Excluded columns: %s" %projected_drop_columns
        exp['Preprocessing'] = "Scaling: %s Dimensionality: %s" %(scaler, dim_red)
        exp['Details'] = "Test Size: %s" %test_size
        exp['Algorithm'] = model.__class__
        exp['Parameters'] = model.get_params()

        # Saving results
        exp['Evaluation'] = "CV F1: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
        exp['F1'] = score
        exp['AUC'] = auc
        exper.save(exp)
        print("Experiment saved")
        
    return model, score

# Actual prediction pipeline
def run_prediction(model, data, predict_prob=False):
    import pandas as pd
    pred_y = model.predict(data)
    if predict_prob:
        pred_y_prob = model.predict_proba(data)[:,1]
        return pd.DataFrame({'label':pred_y, 'probability':pred_y_prob}, index=data.index)
    else:
        return pd.DataFrame({'label':pred_y}, index=data.index)
    
def preprocess_labeled_data(output_column,  dataset=None, output_scaler=None, output_scaler_param=None, reload_data= False, drop_na=True, scaler=None, dim_red=None, feature_extractor=None,  filter_name=None, filter_col_name='AOV', projected_drop_columns=None, projected_select_columns=None, input_exclusion_columns_list=None,   sql_file=None, query_name=None, subset_size=1000, use_all_dataset=False, remove_processed_features=False, trained_scaler=False, trained_dim_red=False):
    
    
    import connect_warehouse, experiments, encoding
    import pandas as pd
    
    if dataset is None or reload_data:
        # Pulling data from DW (You need to write a sql file with the queries)
        print("Pulling data from Warehouse")
        res = connect_warehouse.get_data_using_sql_file(query_name, sql_file)
    else:
        print("Using existing dataset")
        res = dataset.copy()
    
    subset = res
    print("Raw data: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    #Filtering unrelevant datapoints
    if filter_name is not None:
        subset = filter_data(subset, col_name=filter_col_name, filter_name=filter_name)
        print("Filtered: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    # Subsetting for faster computations (if working on prototyping)
    if not use_all_dataset:
        #subset = subset[:subset_size]
        subset = subset.sample(subset_size)
        print("Subsetted: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
      
    print(output_column)
    if output_scaler is not None:
        if output_scaler_param is not None:
            subset['LABEL'] = subset[output_column].apply(lambda x: output_scaler(x, output_scaler_param))
        else:
            subset['LABEL'] = subset[output_column].apply(lambda x: output_scaler(x))
    else:
        subset['LABEL'] = subset[output_column]
    
    print("After Creating Label column: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    
    # Projecttion by selection
    if projected_select_columns is not None:
        select_list = projected_select_columns.copy()
        select_list.append('LABEL')
        #projected_select_columns.append(output_column)
        subset = subset[select_list]
        print("Projected after columns selection: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    
    # Feature extraction, adding calculated columns, pivoting
    if feature_extractor is not None:
        subset, converted_columns = feature_extractor(subset)
        print("After adding features: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
        
    # Projection by elimination - needs to be removed
    if projected_drop_columns is not None:
        subset = subset.drop(projected_drop_columns, axis=1)
        print("Projected after columns removal: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    # Removing original features
    if remove_processed_features and converted_columns is not None:
        subset = subset.drop(converted_columns, axis=1)
        print("Projected after original features removal: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
        print(subset.count())
    
    # Check & removing missing data
    print("Number of datapoints with NaN: %s" %subset[subset.isnull().any(axis=1)].shape[0])

    # drop NaNs
    if drop_na:
        subset = subset.dropna()
        print("After dropping NaNs: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    # Splitting to input and output
    X, Y = split_to_x_y(subset, 'LABEL', input_exclusion_columns_list)
    print("X dimension: %s rows and %s columns" %(X.shape[0], X.shape[1]))
    

    
    
    #Scalilng
    print(scaler)
    print(trained_scaler)
    if scaler is not None:
        vector = X.as_matrix()
        if not trained_scaler:
            scaler = scaler.fit(vector)
        vector_scaled = scaler.transform(vector)
        print("After Scaling Input: %s rows and %s columns" %(vector_scaled.shape[0], vector_scaled.shape[1]))

        #to dataframe (needs testing!!)
        X = pd.DataFrame(vector_scaled, index=subset.index)
    
    #Dimensionality reduction
    if dim_red is not None:
        print("X original dimension is: (%s)" %X.shape[1])
        from sklearn.feature_selection import SelectFromModel
        if not trained_dim_red :
            dim_red = dim_red.fit(X)
        #dim_model = SelectFromModel(reshaper, prefit=True)
        X = dim_red.transform(X)
        print("X new dimension after dimensionality reduction is: (%s)" %X.shape[1])
    
    return X, Y, scaler, dim_red
    
def preprocess_data(dataset=None,reload_data= False, drop_na=True, scaler=None, dim_red=None, feature_extractor=None,  filter_name=None, filter_col_name='AOV', projected_drop_columns=None, projected_select_columns=None, input_exclusion_columns_list=None,   sql_file=None, query_name=None, subset_size=1000, use_all_dataset=False, remove_processed_features=False, trained_scaler=False, trained_dim_red=False):
    
    import connect_warehouse, experiments, encoding
    import pandas as pd
    
    if dataset is None or reload_data:
        # Pulling data from DW (You need to write a sql file with the queries)
        print("Pulling data from Warehouse")
        res = connect_warehouse.get_data_using_sql_file(query_name, sql_file)
    else:
        print("Using existing dataset")
        res = dataset.copy()
    
    subset = res
    print("Raw data: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    #Filtering unrelevant datapoints
    if filter_name is not None:
        subset = filter_data(subset, col_name=filter_col_name, filter_name=filter_name)
        print("Filtered: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    # Subsetting for faster computations (if working on prototyping)
    if not use_all_dataset:
        #subset = subset[:subset_size]
        subset = subset.sample(subset_size)
        print("Subsetted: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    """  
    if output_scaler is not None:
        if output_scaler_param is not None:
            subset['LABEL'] = subset[output_column].apply(lambda x: output_scaler(x, output_scaler_param))
        else:
            subset['LABEL'] = subset[output_column].apply(lambda x: output_scaler(x))
    else:
        subset['LABEL'] = subset[output_column]
    
    print("After Creating Label column: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    """
    
    # Projecttion by selection
    if projected_select_columns is not None:
        select_list = projected_select_columns.copy()
        #select_list.append('LABEL')
        #projected_select_columns.append(output_column)
        subset = subset[select_list]
        print("Projected after columns selection: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    
    # Feature extraction, adding calculated columns, pivoting
    if feature_extractor is not None:
        subset, converted_columns = feature_extractor(subset)
        print("After adding features: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
        
    # Projection by elimination - needs to be removed
    if projected_drop_columns is not None:
        subset = subset.drop(projected_drop_columns, axis=1)
        print("Projected after columns removal: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    
    # Removing original features
    if remove_processed_features and converted_columns is not None:
        subset = subset.drop(converted_columns, axis=1)
        print("Projected after original features removal: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
        print(subset.count())
    
    # Check & removing missing data
    print("Number of datapoints with NaN: %s" %subset[subset.isnull().any(axis=1)].shape[0])

    # drop NaNs
    if drop_na:
        subset = subset.dropna()
        print("After dropping NaNs: %s rows and %s columns" %(subset.shape[0], subset.shape[1]))
    """
    # Splitting to input and output
    X, Y = split_to_x_y(subset, 'LABEL', input_exclusion_columns_list)
    print("X dimension: %s rows and %s columns" %(X.shape[0], X.shape[1]))
    """

    X = subset
    
    #Scalilng
    print(scaler)
    print(trained_scaler)
    if scaler is not None:
        vector = X.as_matrix()
        if not trained_scaler:
            scaler = scaler.fit(vector)
        vector_scaled = scaler.transform(vector)
        print("After Scaling Input: %s rows and %s columns" %(vector_scaled.shape[0], vector_scaled.shape[1]))

        #to dataframe (needs testing!!)
        X = pd.DataFrame(vector_scaled, index=subset.index)
    
    #Dimensionality reduction
    if dim_red is not None:
        print("X original dimension is: (%s)" %X.shape[1])
        from sklearn.feature_selection import SelectFromModel
        if not trained_dim_red :
            dim_red = dim_red.fit(X)
        #dim_model = SelectFromModel(reshaper, prefit=True)
        X = dim_red.transform(X)
        print("X new dimension after dimensionality reduction is: (%s)" %X.shape[1])
    
    return X, scaler, dim_red
    
    
    #---- Plotting

def plot_heatmaps (vis, colv, colh, valuecol, bycol, aggfunc=np.average, figsize=(10, 10)):
    
    import seaborn as sns
    import matplotlib.pyplot as plt

    vis[colv] = vis[colv].astype('category')
    vis[colh] = vis[colh].astype('category')
    vis[bycol] = vis[bycol].astype('category')
    vis[valuecol] = vis[valuecol].astype('f8')

    for i, cat in enumerate(set(vis[bycol])):

        mat = vis[vis[bycol] == cat].pivot_table(index=colv, columns=colh, values=valuecol, aggfunc=aggfunc)

        # Set up the matplotlib figure
        f, ax = plt.subplots(sharey=True, figsize=figsize)

        # Generate a custom diverging colormap
        cmap = sns.diverging_palette(220, 5, as_cmap=True, center="dark")

        # Draw the heatmap with the mask and correct aspect ratio
        sns.heatmap(mat, cmap=cmap, 
                    square=True,
                    linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
        plt.title("""Heatmap of: %s \n Agg_func: %s \n %s: %s \n Support: %s"""
         %(valuecol, aggfunc.__name__, bycol, cat, vis[vis[bycol]==cat].shape[0]))
        plt.show()
    return plt

def plot_heatmap (vis, colv, colh, valuecol, aggfunc=np.average, figsize=(10, 10)):
    
    import seaborn as sns
    import matplotlib.pyplot as plt

    vis[colv] = vis[colv].astype('category')
    vis[colh] = vis[colh].astype('category')
    vis[valuecol] = vis[valuecol].astype('f8')

    mat = vis.pivot_table(index=colv, columns=colh, values=valuecol, aggfunc=aggfunc)

    # Set up the matplotlib figure
    f, ax = plt.subplots(sharey=True, figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(220, 5, as_cmap=True, center="dark")

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(mat, cmap=cmap, 
                square=True,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.title("""Heatmap of: %s \n Agg_func: %s \n Support: %s"""
     %(valuecol, aggfunc.__name__, vis.shape[0]))
    plt.show()
    return plt

def plot_hists(vis, bins=100, figsize=(11, 4)):
    # to draw a histogram for each variable
    import matplotlib.pyplot as plt

    num_vars = len(vis.columns)
    plt.figure(1, figsize=(figsize[0], figsize[1]*num_vars))
    for i, var in enumerate(vis.columns):
        plt.subplot(num_vars,1,1+i)
        try:
            vis[var].hist(bins=100)
        except Exception as e:
            if len(vis[var].unique()) > 100:
                print("Could not plot histogram for variable %s.  Will be ignored" %var)
                print(e)
            else:
                vis[var].value_counts().plot(kind='bar')
                print(e)
        plt.title(var)

def plot_corr_heatmap(data, figsize=(12, 10)):
    # Plot correlation heatmap

    import seaborn as sns
    import matplotlib.pyplot as plt
    
    # Compute the correlation matrix
    corr = data.corr()

    # Generate a mask for the upper triangle
    mask = np.zeros_like(corr, dtype=np.bool)
    mask[np.triu_indices_from(mask)] = True

    # Set up the matplotlib figure
    f, ax = plt.subplots(figsize=figsize)

    # Generate a custom diverging colormap
    cmap = sns.diverging_palette(10, 220, as_cmap=True)

    # Draw the heatmap with the mask and correct aspect ratio
    sns.heatmap(corr, mask=mask, cmap=cmap, vmax=1,
                square=True,
                linewidths=.5, cbar_kws={"shrink": .5}, ax=ax)
    plt.title("Metrics Correlation Matrix")
    return plt

def plot_roc_curve(Y_actual, Y_predicted):
    from sklearn.metrics import roc_curve
    from sklearn.metrics import auc
    from matplotlib import pyplot as plt
    fpr, tpr, threshold = roc_curve(Y_actual, Y_predicted)
    area = auc(fpr, tpr)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.3f)' % area)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (1-Specificity)')
    plt.ylabel('True Positive Rate (Sensitivity - Recall)')
    plt.title('Receiver operating characteristic')
    plt.legend(loc="lower right")
    return plt


def plot_learning_curve(estimator, X, y,title="Learning Curve",ylim=None, cv=None,
                        n_jobs=1, train_sizes=linspace(.1, 1.0, 10), scoring='accuracy'):
    """
    Generate a simple plot of the test and traning learning curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : integer, cross-validation generator, optional
        If an integer is passed, it is the number of folds (defaults to 3).
        Specific cross-validation objects can be passed, see
        sklearn.cross_validation module for the list of possible objects

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    import numpy as np
    from matplotlib import pyplot as plt
    from sklearn.learning_curve import learning_curve
    
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes, scoring=scoring)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


def plot_cv_curve(estimator, X, y, param_name, param_range=linspace(1, 10, 10),
        cv=None, scoring="accuracy", n_jobs=1):
    from sklearn.learning_curve import validation_curve
    import numpy as np
    from matplotlib import pyplot as plt
    
    plt.figure()
    train_scores, test_scores = validation_curve(
        estimator, X, y, param_name=param_name, param_range=param_range,
        cv=cv, scoring=scoring, n_jobs=n_jobs)
    
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title("Validation Curve")
    plt.xlabel(param_name)
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    plt.xlim(np.min(param_range), np.max(param_range))
    plt.semilogx(param_range, train_scores_mean, label="Training score", color="r")
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2, color="r")
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="g")
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2, color="g")
    plt.legend(loc="best")
    return plt

def plot_prec_recall_curve(Y_test, Y_predicted):
    from sklearn.metrics import precision_recall_curve
    import numpy as np
    from matplotlib import pyplot  as plt
    plt.figure()
    precision, recall, thresholds = precision_recall_curve(Y_test, Y_predicted)
    plt.clf()
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    return plt

def plot_prec_recall_by_threshold(Y_test, Y_predicted):
    """ Might need better aligning. Take numbers with a grain of salt"""
    from sklearn.metrics import precision_recall_curve
    import numpy as np
    from matplotlib import pyplot as plt
    precision, recall, thresholds = precision_recall_curve(Y_test, Y_predicted)
    plt.figure()
    plt.plot(thresholds, precision[0:-1], label='Precision')
    plt.plot(thresholds, recall[:-1], label='Recall')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall')
    plt.legend(loc="lower left")
    return plt


def plot_confusion_matrix(con_m, title='Confusion matrix', cmap=pyplot.cm.OrRd_r):
    from matplotlib import pyplot as plt
    import numpy as np
    cm = con_m
    plt.figure()
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    plt.xticks(tick_marks, rotation=0)
    plt.yticks(tick_marks)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    for x in range(2):
        for y in range(2):
            plt.annotate(str(cm[x][y]), xy=(y, x), 
                    horizontalalignment='center',
                    verticalalignment='center')
    return plt

def plot_perf_plots(y_true, y_pred, y_prob=None, title="Experiment"):
    print ("Results for : %s" %title)
    print("") 
    
    from sklearn.metrics import f1_score, classification_report, confusion_matrix
    score = f1_score(y_true, y_pred)
    print("F1-score on test set: %0.3f" % (score))
    print

    print(classification_report(y_true, y_pred))
    print("")
    
    #pyplot.subplot(2,2,1)
    plot_confusion_matrix(confusion_matrix(y_true, y_pred))

    if y_prob is not None: 
        #pyplot.subplot(2,2,2)
        plot_prec_recall_curve(y_true, y_prob)
        #pyplot.subplot(2,2,3)
        plot_prec_recall_by_threshold(y_true, y_prob)
        #pyplot.subplot(2,2,4)
        plot_roc_curve(y_true, y_prob)
        
def plot_aov_perf_plots(data, prob, pos_field, neg_field, bins=10, filtering_field=None, filtering_value=None, stacking=False):
    predictions = data.copy()
    predictions['model_probability'] = prob

    #predictions.index = predictions['model_att_probability']
    if filtering_field is not None:
        if filtering_value is None:
            raise ValueError("Both filtering field name and value needs to be set")
        else:
            predictions = predictions[predictions[filtering_field] < filtering_value]
    
    predictions = predictions.sort(['model_probability'], ascending=False)

    #Binning data by probability threshold
    from numpy import linspace, sum
    from pandas import cut
    binings = linspace(0,1,bins)
    groups = predictions.groupby(cut(predictions.model_probability, binings))
    
    #Alternative way that results in bins with sequencial number
    #groups = predictions.groupby(np.digitize(predictions.model_att_probability, bins))

    #Plotting performance by AOV
    aov_by_p = groups.agg({pos_field:sum, neg_field: sum})
    aov_by_p.plot( kind='bar', stacked=stacking)