from os import environ

AZCFL = 'sfm-pi-db-ld1:1531/azcfl'
BIDWPRD_BU = 'db-bidwprd.internal.salesforce.com:1531/bidwprd_bu'

#Unused
BIDWPRD_BU_OLD = 'sfm-db-edw-lp-c1-scan:1531/bidwprd_bu'

USER = environ['DW_U']
PASS = environ['DW_P']

 # This is defined as a windows ODBC for AZCFL
AZCFL_ODBC='OracleNewDW'
BIDWPRD_ODBC = 'OracleBIDWPRD'


def get_azcfl_cursor(ODBC=False, ODBC_NAME=AZCFL_ODBC):
    if ODBC==True:
        import pyodbc as po
        con = po.connect('DSN=%s;PWD=%s' %(AZCFL_ODBC,PASS))
    else:
        import cx_Oracle
        con = cx_Oracle.connect('%s/%s@%s' %(USER,PASS,AZCFL))
    return con.cursor()

def get_prod_cursor(ODBC=False, ODBC_NAME=BIDWPRD_ODBC):
    if ODBC==True:
        import pyodbc as po
        con = po.connect('DSN=%s;PWD=%s' %(BIDWPRD_ODBC,PASS))
    else:
        import cx_Oracle
        con = cx_Oracle.connect('%s/%s@%s' %(USER,PASS,BIDWPRD_BU))
    return con.cursor()


def get_azcfl_con(ODBC=False, ODBC_NAME=AZCFL_ODBC):
    if ODBC==True:
        import pyodbc as po
        con = po.connect('DSN=%s;PWD=%s' %(AZCFL_ODBC,PASS))
    else:
        import cx_Oracle
        con = cx_Oracle.connect('%s/%s@%s' %(USER,PASS,AZCFL))
    return con

def get_prod_con(ODBC=False, ODBC_NAME=BIDWPRD_ODBC):
    if ODBC==True:
        import pyodbc as po
        con = po.connect('DSN=%s;PWD=%s' %(BIDWPRD_ODBC,PASS))
    else:
        import cx_Oracle
        con = cx_Oracle.connect('%s/%s@%s' %(USER,PASS,BIDWPRD_BU))
    return con


def load_sql_file(path=None):
    """
    Reads a sql file written in the following format and returns a dictionary with all queries:
    # -- some comment here  # comments start with hashtag then space.  Current version does not support sql comments or yaml comments within queries
    get_aov: 
     SELECT *
     FROM monthly_aov_mc
     WHERE acct_id = '00130000004EihB'

    # -- some comments here2
    get_aov_agg: 
     SELECT * 
     FROM monthly_aov_mc_agg 
     WHERE acct_id = '%s'"""
    if path is None:
        raise IOError("No sql_file path was provided")
    import yaml
    f = open(path, 'r')
    return yaml.load(f.read())


def get_query(query=None, path=None):
	if query is None or path is None:
		raise AttributeError("Please specify the query and the path of .sql file")
	else:
		return load_sql_file(path)[query]


"""
Example for using sql statements

cur.execute('select * from departments order by department_id')
               
res = cur.fetchall()
for r in res:
    print r
cur.close()
con.close()

"""

"""
from connect_warehouse import load_sql_file, get_azcfl_con
from pandas import read_sql

queries = load_sql_file("../../common/SQL/YAML_SAMPLE.sql")
con = get_azcfl_con()
data = read_sql(queries['get_aov'], con)
con.close()  #Very important not to forget
"""
