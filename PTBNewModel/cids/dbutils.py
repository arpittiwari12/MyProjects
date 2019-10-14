import pandas as pd
import pandas.io.sql as psql
import configparser
import calendar, datetime as dt
from .constants import *
import sqlalchemy
import logging
import numpy as np

#init logger
logging.basicConfig(level= logging.ERROR, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# import cx_Oracle if it exists
try:
    import cx_Oracle
except:
    logger.warn('Oracle client and/or cx_Oracle installation not found. dbutils will not work')

def fetchDBConnection(dbcred_file = USER_CRED_FILE):
    """Return a cx_Oracle connection to the database defined in dbcred_file

    Parameters
    -----------
    dbcred_file : string
        Path to the database credentials file (typically at $HOME/.dbuser.cred). The contents should like so::

            [database_creds]
            host:<HOST>
            port:<PORT>
            user:<USER>
            database:<DATABASE>
            password:<PASSWORD>
 
    Returns
    --------
    conn : cx_Oracle connection to the database

    """
    conn = cx_Oracle.connect('''{USER}/{PASSWORD}@{HOST}:{PORT}/{DATABASE}'''.format(
                    **fetchDBCredentials(dbcred_file)
                ))
    return conn

def fetchDBCredentials(dbcred_file = USER_CRED_FILE):
    """Read database access credentials from the file in $HOME/.dbuser.cred

    Parameters
    ----------
    dbcred_file : string
        Path to the database credentials file (typically at $HOME/.dbuser.cred). The contents should like so::

            [database_creds]
            host:<HOST>
            port:<PORT>
            user:<USER>
            database:<DATABASE>
            password:<PASSWORD>        

    Returns
    --------
    dbcreds : dict
        A dictionary containing the database credentials defined in ``dbcred_file``

    """
    # Read database credentials from user supplied file
    conf = configparser.ConfigParser()
    conf.read(dbcred_file)
    # host, port, user, database, password
    host = conf.get('database_creds','host')
    port = conf.get('database_creds','port')
    user = conf.get('database_creds','user')
    database = conf.get('database_creds','database')
    password = conf.get('database_creds','password')
    dbcreds = {
                'USER': user, 
                'PASSWORD': password, 
                'DATABASE': database, 
                'HOST': host, 
                'PORT': port
            }   
    return dbcreds     
        
def fetchSQLAlchemyEngine(dbcred_file = USER_CRED_FILE):
    """Create and return a SQL alchemy engine

    Parameters
    ----------
    dbcred_file : string
        Path to the database credentials file (typically at $HOME/.dbuser.cred). The contents should like so::

            [database_creds]
            host:<HOST>
            port:<PORT>
            user:<USER>
            database:<DATABASE>
            password:<PASSWORD>        

    Returns
    --------
    sqlalchemy_engine : A SQL Alchemy engine connection to the database

    """
    conn_str = 'oracle+cx_oracle://{USER}:{PASSWORD}@{HOST}:{PORT}/?service_name={DATABASE}'
    sqlalchemy_engine = sqlalchemy.create_engine(
                                conn_str.format(
                                    **fetchDBCredentials(dbcred_file = dbcred_file)
                                ), 
                                coerce_to_unicode = True
                            )
    return sqlalchemy_engine    


def drop_table_if_exists(table_name, conn):
    """Check database catalog to verify if a table exists and drop it if found

    Parameters
    ----------
    table_name : string
        Name of the table to be dropped (``schema_name.table_name`` or just ``table_name``)
    conn : cx_Oracle connection object
        Typically obtained by invoking ``fetchConnection()`` defined above  

    Returns
    --------
    msg : string
        A message indicating if the table was successfully dropped (if found)

    """
    exists_sql = """
        SELECT
            TABLE_NAME
        FROM
            USER_TABLES
        WHERE
            TABLE_NAME = '{table_name}'
    """.format(
        table_name = table_name
    )
    _df = pd.read_sql(exists_sql, conn)
    msg = 'Table {} does not exist, ignoring DROP'.format(table_name)
    if(_df.shape[0] > 0):
        conn.cursor().execute("""DROP TABLE {table_name} """.format(table_name = table_name))
        conn.commit()
        msg = 'Dropped: {}'.format(table_name)
    return msg

def _varchar_length(max_field_length):
    """Determine appropriate length of varchar field depending on max_field_length

    Parameters
    -----------
    max_field_length : int/float
        maximum size of an object(varchar) field in a pandas dataframe

    Returns
    ---------
    varchar_len : int
        The varchar length to be used while persisting this field in the oracle database

    """
    varchar_len = 1
    # If max_field_length is not empty or not NaN, return as is
    # else return '1'
    if(max_field_length and not np.isnan(max_field_length)):
        varchar_len = max_field_length
    return varchar_len


def saveDFAsTable(
        df,
        table_name,
        schema_name = None,
        if_exists = 'fail'
    ):
    """Save the dataframe as a table in AZPRD in the provided schema

    Parameters
    ----------
    df: Pandas dataframe
        The dataframe to be saved as table
    table_name: string
        The name of the table to save the dataframe into
    schema_name: string, default: public/default schema
        The name of the schema to write the dataframe into, if empty will use default schema
    if_exists: string 
        The action to take if the table already exists in the database ('fail', 'replace', 'append')

    Returns
    -------
    Prints the name of the table the dataframe was written into, if successful.

    """
    # Map all object (dtype) columns to VARCHAR2(LENGTH), otherwise these get stored as CLOB!
    date_cols = list(df.select_dtypes(include = ['datetime64']).columns)
    object_cols = list(set(df.select_dtypes(include = ['dtype']).columns) - set(date_cols))
    date_dict = dict([(col, sqlalchemy.types.DATE) for col in date_cols])
    varchar_dict = dict(
                        [
                            (
                                col, 
                                sqlalchemy.types.VARCHAR(
                                    _varchar_length(df[col].str.len().max())
                                )
                            ) 
                            for col in object_cols
                        ]
                    )
    dtype_dict = {**date_dict, **varchar_dict}

    with fetchSQLAlchemyEngine().connect() as sqlalchemy_conn, sqlalchemy_conn.begin():
        df.to_sql(
            table_name,
            sqlalchemy_conn,
            schema = schema_name,
            if_exists = if_exists,
            index = False,
            chunksize = 1000,
            dtype = dtype_dict
        )

    logger.info(
        """Table {}{}{} created""".format(
                '' if not schema_name else schema_name, 
                '.' if schema_name else '', 
                table_name
            )
        )

    return
