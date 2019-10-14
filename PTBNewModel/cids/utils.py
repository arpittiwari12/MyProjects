from .dbutils import *
import time
from datetime import datetime
import numpy as np
from scipy import stats
from textwrap import wrap
from enum import Enum
from types import SimpleNamespace
import datetime as dt

class EWS_6_0_Segments(object):
    """Mapping of EWS 6.0 segments based on MACROAOVBAND

    Parameters
    ------------
    
    Returns
    --------

    """
    SEGMENTS = SimpleNamespace(
                    **{
                        'LT_10K'  : '<$10K',
                        'BET_10K_TO_50K': '$10K-50K',
                        'BET_50K_TO_200K' : '$50K-200K',
                        'GT_200K' : '$200k+'
                    }
                )
    
    @staticmethod
    def get_segment(snapshot_dt = None, curr_aov = None):
        """Return the EWS-6.0 segment based on the MACROAOVBAND

        Parameters
        -----------

        snapshot_dt : date, default: None
            The date for which the scores are being computed
        curr_aov : float, default: None
            The current spend of the account (as of snapshot_dt) with Salesforce in USD

        Returns
        --------

            A string representing the EWS-6.0 segment for the account

        """
        if (curr_aov >= 200000):
            return EWS_6_0_Segments.SEGMENTS.GT_200K
        elif((curr_aov >= 50000) & (curr_aov < 200000)):
            return EWS_6_0_Segments.SEGMENTS.BET_50K_TO_200K
        elif((curr_aov >= 10000) & (curr_aov < 50000)):
            return EWS_6_0_Segments.SEGMENTS.BET_10K_TO_50K
        else:
            return EWS_6_0_Segments.SEGMENTS.LT_10K

class EWS360Segments(object):
    """Mapping of EWS360 segments based on MACROAOVBAND and FIRSTYEARFLG

    Parameters
    ------------
    
    Returns
    --------

    """
    SEGMENTS = SimpleNamespace(
                    **{
                        'LT_10K_FIRSTYEAR'  : '<$10K_FIRSTYEAR',
                        'LT_10K_NOTFIRSTYEAR' : '<$10K_NOTFIRSTYEAR',
                        'BET_10K_TO_50K_FIRSTYEAR': '$10K-50K_FIRSTYEAR',
                        'BET_10K_TO_50K_NOTFIRSTYEAR': '$10K-50K_NOTFIRSTYEAR',
                        'BET_50K_TO_200K_FIRSTYEAR' : '$50K-200K_FIRSTYEAR',
                        'BET_50K_TO_200K_NOTFIRSTYEAR' : '$50K-200K_NOTFIRSTYEAR',
                        'GT_200K' : '$200k+'
                    }
                )
    
    @staticmethod
    def get_segment(snapshot_dt, first_active_dt, curr_aov):
        """Return the EWS-360 segment based on the MACROAOVBAND and FIRSTYEAR flag

        Parameters
        -----------
        snapshot_dt : date 
            The date for which the scores are being computed
        first_active_dt : date 
            The date when the account was first active
        curr_aov : float
            The current spend of the account (as of snapshot_dt) with Salesforce in USD

        Returns
        --------
        A string representing the EWS-360 segment for the account

        """
        day_diff = (snapshot_dt - first_active_dt) / np.timedelta64(1, 'D')
        if (curr_aov >= 200000):
            # If Account is > 200k, we don't care whether it is FIRSTYEAR or NOTFIRSTYEAR. 
            # they will be considered in the same peer group
            return EWS360Segments.SEGMENTS.GT_200K
        elif((curr_aov >= 50000) & (curr_aov < 200000)):
            if(day_diff <= 365):
                return EWS360Segments.SEGMENTS.BET_50K_TO_200K_FIRSTYEAR
            else:
                return EWS360Segments.SEGMENTS.BET_50K_TO_200K_NOTFIRSTYEAR
        elif((curr_aov >= 10000) & (curr_aov < 50000)):
            if(day_diff <= 365):
                return EWS360Segments.SEGMENTS.BET_10K_TO_50K_FIRSTYEAR
            else:
                return EWS360Segments.SEGMENTS.BET_10K_TO_50K_NOTFIRSTYEAR
        else:
            if(day_diff <= 365):
                return EWS360Segments.SEGMENTS.LT_10K_FIRSTYEAR
            else:
                return EWS360Segments.SEGMENTS.LT_10K_NOTFIRSTYEAR

class EWS360ClassLabels(Enum):
    """Mapping of numeric class indicators to labels 

    Parameters
    ----------

    Returns
    --------
    
    """
    GROWTH = 0
    RETENTION = 1
    PARTIAL_ATTR = 2
    TOTAL_ATTR = 3

class CIPRFClassLabels(Enum):
    """Mapping of numeric class indicators to labels 

    Parameters
    ----------

    Returns
    --------
    
    """
    NO_ATTR = 0
    PARTIAL_ATTR = 1
    TOTAL_ATTR = 2

# Vectorizer to convert an integer label to text (Enum name).
ews360_class_labels_vectorized = np.vectorize(lambda cls: EWS360ClassLabels(cls).name)
ciprf_class_labels_vectorized = np.vectorize(lambda cls: CIPRFClassLabels(cls).name)

def print_wallclock_execution_time(starttime):
    """Print elapsed wall-clock time since ``starttime``

    Parameters
    -----------
    starttime : datetime.time 
        Start time of some unit of code

    Returns
    --------
    current_time : datetime.time
        Prints the time elapsed (days/hours/minutes) since ``starttime`` and returns the current time

    """
    current_time = time.time()
    elapsed_time = int(current_time - starttime)
    sec = dt.timedelta(seconds=int(elapsed_time))
    d = dt.datetime(1,1,1) + sec
    results = None
    if d.day-1 > 0:
        results = 'Wall time: {}d {}h {}m {}s'.format(d.day-1, d.hour, d.minute, d.second)
    elif d.hour > 0:
        results = 'Wall time: {}h {}m {}s'.format(d.hour, d.minute, d.second)
    elif d.minute > 0:
        results = 'Wall time: {}m {}s'.format(d.minute, d.second)
    else:
        results = 'Wall time: {}s'.format(d.second,)
    logger.info(results)
    return current_time