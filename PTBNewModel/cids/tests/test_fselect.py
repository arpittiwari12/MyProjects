from ..fselect import *
import numpy as np
import uuid
import nose.tools
from nose.tools import assert_equals, raises

def test_fetch_multicollinear_features():
    """Test multi-collinearity detection"""
    df = pd.DataFrame(np.random.rand(100, 3), columns = ['x', 'y', 'z'])
    # Create few multi-collinear columns by combining other columns (add some noise to make sure
    # it is not perfect multi-collinearity)
    df['mcol_x'] = df['x'] + np.random.rand(100)*0.10
    df['mcol_yz'] = df['y'] + df['z'] + np.random.rand(100)*0.10
    FEATURE_COLUMNS = sorted(['x', 'y', 'z', 'mcol_x', 'mcol_yz'])
    # Create a dummy string column
    df['ACCOUNT_ID'] = df['x'].apply(lambda x: 'account_{:.3f}'.format(x))

    # I) Test multi-collinearity by explicitly specifying the FEATURE_COLUMNS
    result = fetch_multicollinear_features(
                    df,
                    FEATURE_COLUMNS
                )
    mcf = [c for c,v in result]

    # 1) both x and mcol_x cannot simultaneously be present in the result
    assert_equals(
            ('x' in mcf and 'mcol_x' in mcf),
            False,
            'Multi-collinearity still exists'
        )
    # 2) y, z and mcol_yz cannot simultaneously be present in the result
    assert_equals(
            ('y' in mcf and 'z' in mcf and 'mcol_yz' in mcf),
            False,
            'Multi-collinearity still exists'
        )
    # II) The fetch_multicollinear_features should give same results
    # even if we don't specify the FEATURE_COLUMNS (i.e. it should)
    # detect that ACCOUNT_ID is not a numeric column
    result = fetch_multicollinear_features(df)
    mcf = [c for c,v in result]
    assert_equals(
            ('x' in mcf and 'mcol_x' in mcf),
            False,
            'Multi-collinearity still exists'
        )
    # 2) y, z and mcol_yz cannot simultaneously be present in the result
    assert_equals(
            ('y' in mcf and 'z' in mcf and 'mcol_yz' in mcf),
            False,
            'Multi-collinearity still exists'
        )