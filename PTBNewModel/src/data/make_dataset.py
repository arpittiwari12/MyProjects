import numpy as np
import pandas as pd

import cx_Oracle
import getpass

import configparser
import os

def fetchDBCredentials(dbcred_file):
    """
       Read database access credentials from the file in $HOME/.dbuser.cred
    """
    #Read database credentials from user supplied file
    conf = configparser.ConfigParser()
    conf.read(dbcred_file)
    #host, port, user, database, password
    host = conf.get('database_creds','host')
    port = conf.get('database_creds','port')
    user = conf.get('database_creds','user')
    database = conf.get('database_creds','database')
    password = conf.get('database_creds','password')
    return {'USER':user,'PASSWORD':password,'DATABASE':database,'HOST':host,'PORT':port}

def azcfl_connect():
    '''Connect to AZCFL
    
    Parameters
    ----------
    <None>
    
    Returns
    -------
    cx_Oracle.Connection instance
    '''
    USER_CRED_FILE = os.path.join(os.path.expanduser('~'), '.dbuser.cred')

    con = cx_Oracle.connect('''{USER}/{PASSWORD}@{HOST}:{PORT}/{DATABASE}'''.format(
                **fetchDBCredentials(USER_CRED_FILE)
            ))
    
    return(con)

def load_data_from_query(query):
    '''Connects to AZCFL and reads query results to df
    
    Parameters
    ----------
    query : str
        PL SQL query string
        
    Returns
    -------
    pandas.DataFrame
    '''
    con = azcfl_connect()
    df = pd.read_sql(query, con)
    con.close()
    return(df)

def load_data(table):
    '''Runs simple select statement on table
    
    Parameters
    ----------
    table : str
        Schema and table name to select from in AZCFL
    
    Returns
    -------
    pandas.DataFrame
    '''
    query = 'SELECT * FROM {0}'.format(table)
    return(load_data_from_query(query))

def columns_nulls(df):
    '''Returns list of columns with any nulls
    
    Parameters
    ----------
    df : pandas.DataFrame
        Dataframe in which to find nulls
    
    Returns
    -------
    pandas.Series
        nonzero counts of nulls, indexed by column name
    '''
    nulls = pd.isnull(df).sum()
    return(nulls.loc[nulls > 0])

def drop_null_columns(df, p = 0.1, nc = None):
    '''Drops columns with null % > p 
    
    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to manipulate
    p : float, default 0.1
        percentage null values threshold to
        drop columns
    nc : pandas.Series, optional
        null count series from column_nulls
    
    Returns
    -------
    pandas.DataFrame
        dataframe with null columns dropped
    '''
    if not nc:
        nc = columns_nulls(df)
    pns = nc / len(df.index)
    columns_to_drop = pns.loc[pns > p].index
    return(df.drop(columns_to_drop, axis=1))

def impute_zeros(df, nc = None):
    '''replaces np.NaN with 0 in num columns
    
    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to manipulate
    nc : pandas.Series, optional
        null count series from column_nulls
        
    Returns
    -------
    pandas.DataFrame
        dataframe with 0 imputed
    '''
    if not nc:
        nc = columns_nulls(df)
    num_cols = numeric_columns(df)
    columns_to_impute = nc.loc[num_cols].index
    df[columns_to_impute] = df[columns_to_impute].fillna(0)
    return(df)

def numeric_columns(df):
    '''boolean series with column index, True if numeric
    
    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to manipulate
    
    Returns
    -------
    pandas.Series
        boolean valued, indexed by df.columns, True if numeric
    '''
    num_cols = (df.dtypes == 'float64') | (df.dtypes == 'int64')
    return(num_cols)

def drop_null_rows(df):
    '''Drops rows with any nulls across columns
    
    Parameters
    ----------
    df : pandas.DataFrame
        dataframe to manipulate
    
    Returns
    -------
    df : pandas.DataFrame
        dataframe with null rows dropped
    '''
    nr = pd.isnull(df).sum(axis = 1)
    rows_to_drop = nr.loc[nr > 0].index
    return(df.drop(rows_to_drop))