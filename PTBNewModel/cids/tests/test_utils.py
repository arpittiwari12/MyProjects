from ..utils import *
import numpy as np
import uuid
import nose.tools
from nose.tools import assert_equals, raises

def test_EWS_6_0_Segments():
    """Test generation of EWS 6.0 segment"""
    assert_equals(
            EWS_6_0_Segments.get_segment(
                np.datetime64('2012-12-20'),
                200001
            ),
            EWS_6_0_Segments.SEGMENTS.GT_200K,
            'Segment lookup failed'
        )
    assert_equals(
            EWS_6_0_Segments.get_segment(
                np.datetime64('2012-12-20'),
                100000
            ),
            EWS_6_0_Segments.SEGMENTS.BET_50K_TO_200K,
            'Segment lookup failed'
        )
    assert_equals(
            EWS_6_0_Segments.get_segment(
                np.datetime64('2012-12-20'),
                20000
            ),
            EWS_6_0_Segments.SEGMENTS.BET_10K_TO_50K,
            'Segment lookup failed'
        )
    assert_equals(
            EWS_6_0_Segments.get_segment(
                np.datetime64('2012-12-20'),
                5000
            ),
            EWS_6_0_Segments.SEGMENTS.LT_10K,
            'Segment lookup failed'
        )
    assert_equals(
            EWS_6_0_Segments.get_segment(
                np.datetime64('2012-12-20'),
                200001
            ),
            EWS_6_0_Segments.SEGMENTS.GT_200K,
            'Segment lookup failed'
        )

def test_EWS360Segments():
    """Test generation of EWS 360 segment"""
    assert_equals(
            EWS360Segments.get_segment(
                # Snapshot date
                np.datetime64('2012-12-20'), 
                # First active date
                np.datetime64('2011-12-25'), 
                5000
            ),
            EWS360Segments.SEGMENTS.LT_10K_FIRSTYEAR,
            'Segment lookup failed'
        )    
    assert_equals(
            EWS360Segments.get_segment(
                # Snapshot date
                np.datetime64('2012-12-20'), 
                # First active date
                np.datetime64('2010-12-20'), 
                5000
            ),
            EWS360Segments.SEGMENTS.LT_10K_NOTFIRSTYEAR,
            'Segment lookup failed'
        )   
    assert_equals(
            EWS360Segments.get_segment(
                # Snapshot date
                np.datetime64('2012-12-20'), 
                # First active date
                np.datetime64('2011-12-25'), 
                40000
            ),
            EWS360Segments.SEGMENTS.BET_10K_TO_50K_FIRSTYEAR,
            'Segment lookup failed'
        )   
    assert_equals(
            EWS360Segments.get_segment(
                # Snapshot date
                np.datetime64('2012-12-20'), 
                # First active date
                np.datetime64('2010-12-20'), 
                40000
            ),
            EWS360Segments.SEGMENTS.BET_10K_TO_50K_NOTFIRSTYEAR,
            'Segment lookup failed'
        )   
    assert_equals(
            EWS360Segments.get_segment(
                # Snapshot date
                np.datetime64('2012-12-20'), 
                # First active date
                np.datetime64('2011-12-25'), 
                60000
            ),
            EWS360Segments.SEGMENTS.BET_50K_TO_200K_FIRSTYEAR,
            'Segment lookup failed'
        )  
    assert_equals(
            EWS360Segments.get_segment(
                # Snapshot date
                np.datetime64('2012-12-20'), 
                # First active date
                np.datetime64('2010-12-20'), 
                60000
            ),
            EWS360Segments.SEGMENTS.BET_50K_TO_200K_NOTFIRSTYEAR,
            'Segment lookup failed'
        )   
    assert_equals(
            EWS360Segments.get_segment(
                # Snapshot date
                np.datetime64('2012-12-20'), 
                # First active date
                np.datetime64('2010-12-20'), 
                200001
            ),
            EWS360Segments.SEGMENTS.GT_200K,
            'Segment lookup failed'
        )    

def test_print_wallclock_execution_time():
    """Test wallclock execution time function"""
    current_time = time.time()
    result_time = print_wallclock_execution_time(current_time)
    assert_equals(
            result_time > current_time,
            True,
            'print_wallclock_execution_time failed'
        )