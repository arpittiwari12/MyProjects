from ..dbutils import *
import numpy as np
import uuid
import nose.tools
from nose.tools import assert_equals, raises

def test_fetchDBCredentials():
    """Test fetching DB credentials from $HOME/.dbuser.cred"""
    creds = fetchDBCredentials()
    assert_equals(
            type(creds) == type(dict()),
            True,
            'Could not fetch database credentials'
        )
    assert_equals(
            sorted(creds.keys()) == sorted(['USER', 'PASSWORD', 'DATABASE', 'HOST', 'PORT']),
            True,
            'Database credentials file does not have all required fields'
        )

def test_fetchDBConnection():
    """Test establishing DB Connection"""
    conn = fetchDBConnection()
    _df = pd.read_sql("""SELECT * FROM DUAL""", conn)
    assert_equals(
            _df.shape[0] > 0, 
            True, 
            'Could not establish connection to the database'
        )

def test_drop_table_if_exists():
    """Test if we're able to drop a table """
    RANDOM_TABLE_NAME = 'cids_{}'.format(str(uuid.uuid4()).replace('-', '_')[:30-len('cids_')]).upper()
    expected_result = 'Table {} does not exist, ignoring DROP'.format(RANDOM_TABLE_NAME)
    result = drop_table_if_exists(RANDOM_TABLE_NAME, fetchDBConnection())
    assert_equals(
            result,
            expected_result,
            'Could not drop non-existing table'
        )

def test_SaveDFAsTable():
    """Test function to save a pandas dataframe to the database"""
    RANDOM_TABLE_NAME = 'cids_{}'.format(str(uuid.uuid4()).replace('-', '_')[:30-len('cids_')]).upper()
    _df = pd.DataFrame(
            [
                ['X', np.datetime64('2012-12-20'), '200000'],
                ['Y', np.datetime64('2013-12-20'), '300000'],
                ['Z', np.datetime64('2014-12-20'), '400000']
            ], 
            columns = [
                    'ACCOUNT_ID', 
                    'SNASHOT_DT', 
                    'AOV'
                ]
            )

    saveDFAsTable(_df, RANDOM_TABLE_NAME)
    conn = fetchDBConnection()
    _df = pd.read_sql("""SELECT * FROM {}""".format(RANDOM_TABLE_NAME), conn)
    assert_equals(
            _df.shape[0] == 3,
            True,
            'Could save dataframe to database table'
        )
    # Clean-up
    drop_table_if_exists(RANDOM_TABLE_NAME, conn)
    # Assert that the table has been dropped
    _df = pd.read_sql(
                """
                    SELECT 
                        * 
                    FROM 
                        USER_TABLES 
                    WHERE 
                        TABLE_NAME = '{}'
                """.format(
                    RANDOM_TABLE_NAME
                ), 
                conn
            )
    assert_equals(
            _df.shape[0] == 0,
            True,
            'Could not clean-up the intermediate table'
        )

