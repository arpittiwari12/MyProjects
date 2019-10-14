"""Feature selection """
# Authors: Srivatsan Ramanujam <srivatsan.ramanujam@salesforce.com>
#
# License: All rights owned by Customer Intelligence, Salesforce.com

from .utils import *
import sklearn
from sklearn import linear_model
from collections import defaultdict
from statsmodels.stats.outliers_influence import variance_inflation_factor
import copy

def fetch_multicollinear_features(
        df,
        FEATURE_COLUMNS = None,
        VIF_THRESHOLD = 10.0
    ):
    """Detect multi-collinearity using VIF (Variance Inflation Factor)

    Refer to https://en.wikipedia.org/wiki/Variance_inflation_factor for more information.

    Parameters
    -----------
    df : Pandas dataframe
        The input dataframe containing a list of attributes including ``FEATURE_COLUMNS`` a subset of which 
        could be multi-collinear (i.e. one or more columns could be re-expressed as a linear combination of others)
    FEATURE_COLUMNS : list, default:  None
        A list of columns in ``df`` that are your independent variables/features. If unspecified or empty, 
        all numeric columns in ``df`` will be selected
    VIF_THRESHOLD : float, default: 10.0
        The threshold to be used to decide if an attritbute is multi-collinear. The higher this value, 
        the stronger the condition that is used to decide if an attribute is multi-collinear

    Returns
    --------
    multicollinear_features : list of tuples
        A list of attributes (a subset of ``FEATURE_COLUMNS``) that were determined to be multi-collinear.
        These attributes may be dis-regarded while training models on ``df``. Every item of this has
        the name of the column in ``FEATURE_COLUMNS`` as the first element and its ``VIF`` as the second element.

    """
    if not FEATURE_COLUMNS:
        FEATURE_COLUMNS = list(df.select_dtypes(include=[np.number]).columns.values)

    multicollinear_features = []
    FEATURE_COLUMNS_COPY = copy.deepcopy(FEATURE_COLUMNS)
    MULTICOLLINEARITY_EXISTS = True
    startIdx = 0
    nextFeat = None
    while MULTICOLLINEARITY_EXISTS:
        logger.info(
            'Step {} of potentially {}'.format(
                    len(FEATURE_COLUMNS) - len(FEATURE_COLUMNS_COPY) + 1, 
                    len(FEATURE_COLUMNS)
                )
            )
        feat_to_remove = None
        if(nextFeat):
            startIdx = FEATURE_COLUMNS_COPY.index(nextFeat)
        for i in range(startIdx, len(FEATURE_COLUMNS_COPY)):
            f = FEATURE_COLUMNS_COPY[i]
            vif = variance_inflation_factor(df[FEATURE_COLUMNS_COPY].values, i)
            # Remove features with high variance inflation factor
            if(vif >= VIF_THRESHOLD):
                logger.info('Discarding feature due to multi-collinearity: {}, VIF: {}'.format(f, vif))
                multicollinear_features.append([f, vif])
                feat_to_remove = f
                nextFeat = FEATURE_COLUMNS_COPY[i+1] if i < len(FEATURE_COLUMNS_COPY)-1 else None
                break
            # If we've iterated through all remaining numeric features and none have high VIF, end
            if(i == len(FEATURE_COLUMNS_COPY)-1):
                MULTICOLLINEARITY_EXISTS = False
        # Remove the feature which is linearly dependent on other features
        if(feat_to_remove):
            FEATURE_COLUMNS_COPY.remove(feat_to_remove)

    return multicollinear_features

