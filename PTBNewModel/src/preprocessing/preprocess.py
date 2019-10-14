import numpy as np
import pandas as pd

import sys
sys.path.append('..')
from data import make_dataset as md

MODEL_VARIABLES = [
    'ACCOUNT_INDUSTRY_NM',
    'EMPLOYEE_COUNT',
    'LOGACTIVEAGE',
    'AOV_BAND',
    'SUPPORT_LEVEL_NM',
    'LIC_NUM_SFDC',
    'LIC_NUM_USR_SFDC',
    'LIC_NUM_USR_SFDC_14D',
    'LIC_UTIL_OEM_ADJ_CUST',
    'LIC_TLP_14D_SFDC',
    'LIC_TLP_14D_CUST',
    'LOG_DIST_CAMPAIGNS_USRS',
    'LOG_DIST_EVENTS_UI_USRS',
    'LOG_DIST_TASKS_UI_USRS',
    'LOG_DIST_CUST_OBJ_UI_USRS',
    'LOG_TOT_RPT_DSHBRD_VWS_PR_USR',
    'LOG_DIST_DSHBRD_USR_VIEWS',
    'LOG_DIST_QUOTES_UI_USRS',
    'LOG_DIST_OPPTY_USRS',
    'LOG_DIST_CASES_FEED_USRS',
    'LOG_DIST_ENTITLEMENTS_UI_USRS',
    'LOG_DIST_CASES_UI_USRS',
    'LOG_DIST_SERVICE_CONSOLE_USRS',
    'LOG_DIST_CASES_API_USRS',
    'LOG_DIST_SOLUTIONS_UI_USRS',
    'LOG_DIST_KNWLDG_SEARCH_USRS',
    'LOG_DIST_CTI_USRS',
    'LOG_DIST_SOAP_API_USRS',
    'ORG_OPPT_SHARING_RULES_NUM',
    'ORG_OPPTS_NUM',
    'ORG_LEADS_PER_USER_NUM',
    'ORG_LEADS_NUM',
    'ORG_PRODUCTS_NUM',
    'ORG_WORKFLOW_RULES_NUM',
    'ORG_CUSTOM_APPS_NUM',
    'ORG_ACCTS_NUM',
    'ORG_CONTACTS_NUM',
    'ORG_ACTIVITIES_NUM',
    'ORG_CUSTOM_OBJECT_RECORDS_NUM',
    'ORG_CASE_QUEUES_NUM',
    'ORG_CASE_REC_TYPES_NUM',
    'ORG_CASE_SHARING_RULES_NUM',
    'ORG_CASES_NUM',
    'ORG_SOLUTIONS_NUM',
    'ORG_APEX_LOC_WRITTEN',
    'CR_SETUP',
    'CO_PHONE',
    'CO_CHAT',
    'CS_OOS',
    'M_ESC_COMM_BIN_CNT',
    'SLA1_MISS_CNT',
    'NUM_MNTH_KA',
    'NUM_MNTH_TRAIL',
    'NUM_MNTH_VIDEO',
    'NUM_MNTH_COMMUNITY',
    'NUM_MNTH_WEBINAR',
    'NUM_MNTH_CSR',
    'LOW_CSAT_CASES_PCT',
    'TOTAL_ACV',
    'OPENPIPE_FTM',
    'PIPEGEN_TTM',
    'LOSTPIPE_TTM',
    'GROWTH_PERCENTAGE_TTM',
    'WIN_PERCENTAGE_TTM',
    'ATTRITION_PERCENTAGE_TTM',
    'TOTAL_OPTIES'
    #'RED_ACCOUNTS',
    #'RED_ACCOUNTS_OPEN_LAST12M'
]

ORDERED_ENCODINGS = {
    'ACCOUNT_TYPE' : {
        'Attrited Customer' : 1,
        'Base Edition' : 1,
        'Contact Manager Edition' : 1,
        'Enterprise Customer' : 3,
        'Group Customer' : 1,
        'Other' : 1,
        'Performance Edition' : 4,
        'Professional Customer' : 2,
        'Prospect' : 1,
        'Unlimited Customer' : 4,
    }
}

def impute_flag(df, outcome = "FLAG"):
    '''added flag inplace
    
    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to add flag, must have
        'PTBFLAG'
    
    Returns
    -------
    <None>
    '''
    df[outcome] = 0
    
    ix = (df['PTBFLAG'] == 1)
    
    df.loc[ix, outcome] = 1
    
def ordered_var_from_dict(series, d):
    '''Creates numeric Series from dict of encodings
    
    Throws KeyError if value in series is not found
    in dict of encodings.
    
    Parameters
    ----------
    series : pandas.Series
        Series of ordered str or object dtype
    d : dict
        dict of encodings, e.g. {"Low": 1, "High": 2}
        
    Returns
    -------
    pandas.Series
        numeric series of ordered encodings
    '''
    ordered_series = pd.Series(index = series.index)
    
    diff = set(series) - set(d.keys())
    if diff:
        raise KeyError(
            "{0} found in data, but not encoding dict".format(diff))

    for k in d.keys():
        ordered_series.loc[series == k] = d[k]
        
    return(ordered_series)

def order_feature(df, column_name,ordered_name = None,
                  encodings = ORDERED_ENCODINGS):
    '''Creates new ordered feature in df inplace
    
    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to manipulate
    column_name : str
        column in df to order, also in ordered_encodings
    ordered_name : str, optional
        name of new column
        Defaults to column_name + "_N"
    
    Returns
    -------
    <None>
    '''
    if not ordered_name:
        ordered_name = "{0}_N".format(column_name)
        
    series = df[column_name]
    d = encodings[column_name]
    
    df[ordered_name] = ordered_var_from_dict(series, d)
    
def dummy_concat(df, column_name):
    '''Adds columns of dummy variables to df
    
    New columns for each level are labeled:
        "{column_name}[T.{level}]"
    
    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to manipulate
    column_name : str
        single column of factors in df to expand
        into dummy variables
        
    Returns
    -------
    pandas.DataFrame
        new dataframe with coded columns
    '''
    dummies = pd.get_dummies(df[column_name])
    dummies.columns = \
    ["{0}[T.{1}]".format(column_name, c) for c in dummies.columns]
    return(dummies)

def subset_encode_df_jorge(df, variables = MODEL_VARIABLES, outcome = 'FLAG'):
    '''Returns df with col subset and dummy encodings
    
    Parameters
    ----------
    df : pandas.DataFrame
        input dataframe to manipulate
    variables : list of str
        list of columns to include in output df
    
    Returns
    -------
    pandas.DataFrame
        output dataframe with encodings
    '''
    # subset df
    #mdf = df[[outcome] + variables]    
    mdf = df
    
    # drop rows with nulls 
    #mdf = md.drop_null_rows(mdf[variables])
    nr = pd.isnull(mdf[variables]).sum(axis = 1)
    rows_to_drop = nr.loc[nr > 0].index
    mdf.drop(rows_to_drop)
    
    # find non-numeric columns
    num_cols = md.numeric_columns(mdf[variables])
    nonnum_cols = list(num_cols.loc[np.logical_not(num_cols)].index)
    
    # create list of encoded variable dataframes
    encodings = []
    for col in nonnum_cols:
        encodings.append(dummy_concat(mdf, col))

    # to append new dummy vbles to the original vbles
    for item in encodings:
        for items in item.columns.values:
            variables.append(items)
    variables.append(outcome)
    
    # to remove non numeric vbles (already dummy created)    
    for item in nonnum_cols:
        variables.remove(item) 
    
    # return concatenation
    return(pd.concat([mdf] + encodings, axis = 1), variables) 


def subset_encode_df(df, variables = MODEL_VARIABLES, outcome = 'FLAG'):
    '''Returns df with col subset and dummy encodings
    
    Parameters
    ----------
    df : pandas.DataFrame
        input dataframe to manipulate
    variables : list of str
        list of columns to include in output df
    
    Returns
    -------
    pandas.DataFrame
        output dataframe with encodings
    '''
    # subset df
    mdf = df[[outcome] + variables]
    
    # drop rows with nulls
    mdf = md.drop_null_rows(mdf)
    
    # find non-numeric columns
    num_cols = md.numeric_columns(mdf)
    nonnum_cols = list(num_cols.loc[np.logical_not(num_cols)].index)
    
    # create list of encoded variable dataframes
    encodings = []
    for col in nonnum_cols:
        encodings.append(dummy_concat(mdf, col))
    
    # drop old columns
    mdf = mdf.drop(nonnum_cols, axis = 1)
    
    # return concatenation
    return(pd.concat([mdf] + encodings, axis = 1))