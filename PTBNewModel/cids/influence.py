"""Compute attribute and dimensional influence in non-linear models"""
# Authors: Srivatsan Ramanujam <srivatsan.ramanujam@salesforce.com>
# License: All rights owned by Customer Intelligence, Salesforce.com

from .utils import *
from operator import itemgetter
import itertools
import multiprocessing
from multiprocessing import Pool
import sklearn
from sklearn import linear_model, decomposition
import xgboost
from collections import defaultdict
import datetime
from scipy import stats

def preprocess_compute_dimensional_influence(
        dims,
        df_scoring,
        FEATURE_COLUMNS,
        xgb_classifier_mdl,
        ClassifierLabel,
        ID_COLUMN = 'ACCOUNT_ID',
        SNAPSHOT_DT_COLUMN = 'SNAPSHOT_DT',
        SEGMENT_COLUMN = 'EWS360_SEGMENT'
    ):
    """Pre-processing routine to speed-up dimensional influence computation.
    
    Parameters
    -----------
    dims : Pandas dataframe
        A dataframe mapping various dimensions, attributes. Should have columns
        ``METRIC``, ``DIMENSION``, ``SUBDIMENSION`` 
    df_scoring : Pandas dataframe
        The dataframe to be scored
    FEATURE_COLUMNS : list
        A sorted list of independent attributes
    xgb_classifier_mdl : XGBClassifier instance 
        An unpickled, pre-trained, multi-class XGBClassifier instance that is trained on ``FEATURE_COLUMNS``
    ClassifierLabel : 
        An Enum denoting the class labels the classifier is trained on . For example:: 

            class ClassifierLabel(Enum):
                GROWTH = 0
                RETENTION = 1
                PARTIAL_ATTR = 2
                TOTAL_ATTR = 3

    ID_COLUMN: string, default : ``ACCOUNT_ID``
        The column in ``df_scoring`` that corresponds to an identifier (ex: ``ACCOUNT_ID`` or ``OPTY_ID`` etc.)
    SNAPSHOT_DT_COLUMN : string, default : ``SNAPSHOT_DT``
        The column in ``df_scoring`` that corresponds to a the snapshot date. 
        Dimensional influence is computed within (``SNAPSHOT_DT_COLUMN``, ``SEGMENT_COLUMN``) groups.
    SEGMENT_COLUMN : string, default : ``EWS360_SEGMENT``
        The column in ``df_scoring`` that corresponds to the account segment.
        Dimensional influence is computed within (``SNAPSHOT_DT_COLUMN``, ``SEGMENT_COLUMN``) groups.

    Returns
    --------
    dim_attr_dict : dict
        A dict mapping of dimensions to attributes
    df_preds : Pandas dataframe 
        ``df_scoring`` input dataframe appended with predictions ``PRED_CLASS`` and ``PRED_PROBA``
    peer_indices_dict :  dict
        A mapping of indices in df_preds that corresponds to peers of each account
    pind_dict_size_df : dataframe
        The size of the peer groups for each  segment
        
    """
    classifier_label_vectorized = np.vectorize(lambda cls: ClassifierLabel(cls).name)

    # dimension to attribute list mapping
    dim_attr_dict = {}
    for d in set(dims['DIMENSION']):
        dim_attr_dict[d] = dims[dims['DIMENSION'] == d]['METRIC'].values

    # The ENGAGEMENT dimension is further divided into PROG_ENGAGEMENT and SUPPORT
    # We will add them as well into our dimensions
    for d in set(dims['SUBDIMENSION']):
        if(d not in dim_attr_dict):
            dim_attr_dict[d] = dims[dims['SUBDIMENSION'] == d]['METRIC'].values

    preds = xgb_classifier_mdl.predict_proba(df_scoring[FEATURE_COLUMNS])

    preds_df = pd.DataFrame(
                            list(
                                zip(
                                    np.max(preds, axis = 1),
                                    classifier_label_vectorized(np.argmax(preds, axis = 1))
                                )
                            ),
                            columns = [
                                'PRED_PROBA',
                                'PRED_CLASS'
                            ],
                            index = df_scoring.index
                        )
    df_preds = pd.concat([df_scoring, preds_df], axis = 1)
    # For every (EWS360_SEGMENT, SNAPSHOT_DT) tuple create a list of indices of 'peers'
    # i.e. index of elements which have the same (EWS360_SEGMENT, SNAPSHOT_DT)
    peer_indices_dict = {}
    for seg, snap_dt, indx in zip(
                                    df_preds[SEGMENT_COLUMN], 
                                    df_preds[SNAPSHOT_DT_COLUMN], 
                                    df_preds.index
                                ):
        # Not using use defaultdict as it is not pickle'able while using multiprocessing
        if(seg not in peer_indices_dict):
            peer_indices_dict[seg] = {snap_dt: [indx]}
        elif(snap_dt not in peer_indices_dict[seg]):
            peer_indices_dict[seg][snap_dt] = [indx]
        else:
            peer_indices_dict[seg][snap_dt].append(indx)

    # Sub-sample peer-indices for performance
    pind_dict_size_lst = []
    for seg in peer_indices_dict:
        for snap_dt in peer_indices_dict[seg]:
            indices = peer_indices_dict[seg][snap_dt]
            # Sample of size 1000 from peer indices
            peer_indices_dict[seg][snap_dt] = np.random.choice(indices, size = min(1000, len(indices)))
            df_segment_shape = df_preds[
                                    (df_preds[SNAPSHOT_DT_COLUMN] == snap_dt) & 
                                    (df_preds[SEGMENT_COLUMN] == seg)
                                ].shape
            pind_dict_size_lst.append([snap_dt, seg, df_segment_shape])

    pind_dict_size_df = pd.DataFrame(
                                sorted(pind_dict_size_lst, key = itemgetter(0, 1)),
                                columns = [ 
                                    SNAPSHOT_DT_COLUMN,
                                    SEGMENT_COLUMN,
                                    'NUM_PEERS'
                                ]
                            )
    return dim_attr_dict, df_preds, peer_indices_dict, pind_dict_size_df

def compute_dimensional_influence_sequential(
        xgb_classifier_mdl, 
        ClassifierLabel,
        ClassifierLabel_best_outcome,
        ClassifierLabel_worst_outcome,
        df_preds,
        FEATURE_COLUMNS,
        dim_attr_dict,
        peer_indices_dict,
        ID_COLUMN = 'ACCOUNT_ID',
        SNAPSHOT_DT_COLUMN = 'SNAPSHOT_DT',
        SEGMENT_COLUMN = 'EWS360_SEGMENT',
        persist_results = True
    ):
    """Computing dimensional influence on the test set using pre-trained model

    Parameters
    -----------
    xgb_classifier_mdl : XGBClassifier 
        Pre-trained XGBClassifier model
    ClassifierLabel : class that extends enum.Enum
        An Enum denoting the class labels the classifier is trained on . For example :: 
        
            class ClassifierLabel(Enum):
                GROWTH = 0
                RETENTION = 1
                PARTIAL_ATTR = 2
                TOTAL_ATTR = 3

    ClassifierLabel_best_outcome : enum 'ClassifierLabel'
        Amongst the labels used in the multi-class classifier, what is the best outcome?
        Example: ``ClassifierLabel.GROWTH``
    ClassifierLabel_worst_outcome : enum 'ClassifierLabel'
        Amongst the labels used in the multi-class classifier, what is the worst outcome?
        Example: ``ClassifierLabel.TOTAL_ATTR``
    df_preds : Pandas dataframe
        A dataframe containing IDs, attributes and predictions  
        should contain columns ``ACCOUNT_ID``, ``EWS360_SEGMENT``, ``SNAPSHOT_DT``, 
        ``PRED_CLASS``, ``PRED_PROBA`` and a list of attributes (``FEATURE_COLUMNS``) used for prediction 
    FEATURE_COLUMNS : list
        Attribute names in df that were used to train xgb_classifier_mdl
    dim_attr_dict : dict
        A dict of dimension to attribute list mapping
    peer_indices_dict : dict
        A dict containing mapping of instance indices belonging to same (peer group, snapshot_dt) pairs
    ID_COLUMN: string, default : ``ACCOUNT_ID``
        The column in ``df_scoring`` that corresponds to an identifier (ex: ``ACCOUNT_ID`` or ``OPTY_ID`` etc.)
    SNAPSHOT_DT_COLUMN : string, default : ``SNAPSHOT_DT``
        The column in ``df_scoring`` that corresponds to a the snapshot date. 
        Dimensional influence is computed within (``SNAPSHOT_DT_COLUMN``, ``SEGMENT_COLUMN``) groups.
    SEGMENT_COLUMN : string, default : ``EWS360_SEGMENT``
        The column in ``df_scoring`` that corresponds to the account segment.
        Dimensional influence is computed within (``SNAPSHOT_DT_COLUMN``, ``SEGMENT_COLUMN``) groups.
    persist_results : boolean, default : True
        Save intermediate results to ``SCRATCHPAD_DATA_HOME`` to measure progress

    Returns
    --------
    result_df : Pandas dataframe
        A dataframe containing predictions and dimensional influence for every instance df

    """
    classifier_label_vectorized = np.vectorize(lambda cls: ClassifierLabel(cls).name)
    # Compute the influence score for each dimension for each instance
    dim_influence_dict = defaultdict(lambda: defaultdict())
    for i in df_preds.index:
        segment, snapshot_dt = df_preds.ix[i][[SEGMENT_COLUMN, SNAPSHOT_DT_COLUMN]].values
        df_preds_copy = df_preds.ix[peer_indices_dict[segment][snapshot_dt]].copy()
        for dim, dim_attr_lst in dim_attr_dict.items():
            # Replacing feature values of the dimension, of the entire test set with the current i'th row values
            df_preds_copy[dim_attr_lst] = df_preds.ix[i][dim_attr_lst].values
            # Calculating the class and probability for the modified test set
            xgb_classifier_y_prob = xgb_classifier_mdl.predict_proba(df_preds_copy[FEATURE_COLUMNS])
            # Majority class for each instance
            pred_classes = classifier_label_vectorized(np.argmax(xgb_classifier_y_prob, axis = 1)) 
            # Probability of majority class for each instance
            pred_probas = np.max(xgb_classifier_y_prob, axis = 1)
            # Select only those instances where majority class is the predicted class for X_test.ix[i]
            filtered_indices = np.where(pred_classes == df_preds.ix[i]['PRED_CLASS'])[0]
            # RE-VISIT: If the predicted classes for the modified dataframe does not match the predicted class
            # for the instance in the original dataset, assign equal score to all dimensions
            dim_influence_i = None
            if filtered_indices.shape[0] > 0:
                dim_influence_i = np.mean(pred_probas[filtered_indices])  
            else:
                dim_influence_i = 1.0/len(dim_attr_dict.keys())
            dim_influence_dict[i]['{}_INF'.format(dim)] = dim_influence_i
            # To compute directionality of influence, we also need to obtain avg(P_growth), avg(P_total_attrition)
            mean_proba_i = np.mean(xgb_classifier_y_prob, axis = 0)
            best_outcome_mean_i = mean_proba_i[ClassifierLabel_best_outcome.value]
            worst_outcome_mean_i =  mean_proba_i[ClassifierLabel_worst_outcome.value]

            dim_influence_dict[i][
                    '{DIM}_INF_{CLS}'.format(
                        DIM = dim, 
                        CLS = ClassifierLabel_best_outcome.name
                    )
                ] = best_outcome_mean_i

            dim_influence_dict[i][
                    '{DIM}_INF_{CLS}'.format(
                        DIM = dim, 
                        CLS = ClassifierLabel_worst_outcome.name
                    )
                ] = worst_outcome_mean_i
    
    # Compile results into a dataframe
    dim_influence_cols = list(
                            dim_influence_dict[
                                list(dim_influence_dict.keys())[0]
                            ].keys()
                        )

    dim_influence_df = pd.DataFrame(
                            [
                                [
                                    dim_influence_dict[i][d] for d in dim_influence_cols
                                ] for i in dim_influence_dict
                            ],
                            columns = dim_influence_cols,
                            index = dim_influence_dict.keys()
                        )

    RESULT_COLUMNS = [
                        ID_COLUMN, 
                        SEGMENT_COLUMN, 
                        SNAPSHOT_DT_COLUMN, 
                        'PRED_PROBA', 
                        'PRED_CLASS'
                    ] + dim_influence_cols

    result_df = pd.concat(
                    [
                        df_preds.ix[dim_influence_df.index], 
                        dim_influence_df
                    ], 
                    axis = 1
                )[RESULT_COLUMNS]
    
    # The unique segments and snap dates in this dataset
    segs = list(set(result_df[SEGMENT_COLUMN]))
    snaps = list(set(result_df[SNAPSHOT_DT_COLUMN]))
    # If called via multiprocessing, save resuls to scratchpad so that progress can be monitored
    if(persist_results):
        filename =  os.path.join(
                            SCRATCHPAD_DATA_HOME,
                            'dim_influence_{}_{}_{}.csv'.format(
                                    str(datetime.datetime.now()).replace(' ', '_'),
                                    segs[0], 
                                    snaps[0].strftime('%Y-%m-%d')
                                )
                        )
        result_df.to_csv(filename, index = False)
        logger.info('Persisting dimensional influence scores to {}'.format(filename))        
    return result_df

def compute_dimensional_influence_parallel(
        xgb_classifier_mdl, 
        ClassifierLabel,
        ClassifierLabel_best_outcome,
        ClassifierLabel_worst_outcome,
        df_preds,
        FEATURE_COLUMNS,
        dim_attr_dict,
        peer_indices_dict,
        ID_COLUMN = 'ACCOUNT_ID',
        SNAPSHOT_DT_COLUMN = 'SNAPSHOT_DT',
        SEGMENT_COLUMN = 'EWS360_SEGMENT',
        persist_results = True,
        n_parallel = int(multiprocessing.cpu_count()*0.50)
    ):
    """Computing dimensional influence on the test set using pre-trained model in parallel

    Parameters
    -----------
    xgb_classifier_mdl : XGBClassifier 
        Pre-trained XGBClassifier model
    ClassifierLabel : class that extends enum.Enum
        An Enum denoting the class labels the classifier is trained on. For example :: 
        
            class ClassifierLabel(Enum):
                GROWTH = 0
                RETENTION = 1
                PARTIAL_ATTR = 2
                TOTAL_ATTR = 3
            
    ClassifierLabel_best_outcome : enum 'ClassifierLabel'
        Amongst the labels used in the multi-class classifier, what is the best outcome?
        Example: ``ClassifierLabel.GROWTH``
    ClassifierLabel_worst_outcome : enum 'ClassifierLabel'
        Amongst the labels used in the multi-class classifier, what is the worst outcome?
        Example: ``ClassifierLabel.TOTAL_ATTR``
    df_preds : Pandas dataframe
        A dataframe containing IDs, attributes and predictions  
        should contain columns ``ACCOUNT_ID``, ``EWS360_SEGMENT``, ``SNAPSHOT_DT``, 
        ``PRED_CLASS``, ``PRED_PROBA`` and a list of attributes (``FEATURE_COLUMNS``) used for prediction 
    FEATURE_COLUMNS : list
        Attribute names in df that were used to train xgb_classifier_mdl
    dim_attr_dict : dict
        A dict of dimension to attribute list mapping
    peer_indices_dict : dict
        A dict containing mapping of instance indices belonging to same (peer group, snapshot_dt) pairs
    ID_COLUMN: string, default : ``ACCOUNT_ID``
        The column in ``df_scoring`` that corresponds to an identifier (ex: ``ACCOUNT_ID`` or ``OPTY_ID`` etc.)
    SNAPSHOT_DT_COLUMN : string, default : ``SNAPSHOT_DT``
        The column in ``df_scoring`` that corresponds to a the snapshot date. 
        Dimensional influence is computed within (``SNAPSHOT_DT_COLUMN``, ``SEGMENT_COLUMN``) groups.
    SEGMENT_COLUMN : string, default : ``EWS360_SEGMENT``
        The column in ``df_scoring`` that corresponds to the account segment.
        Dimensional influence is computed within (``SNAPSHOT_DT_COLUMN``, ``SEGMENT_COLUMN``) groups.
    persist_results : boolean, default : True
        Save intermediate results to ``SCRATCHPAD_DATA_HOME`` to measure progress
    n_parallel = int, default : int(multiprocessing.cpu_count()*0.50)
        The number of workers to run the processing on (dependency on number of CPU cores available)
    
    Returns
    --------
    result : list of Pandas dataframes
        A list of dataframes containing predictions and dimensional influence for every instance in df_preds 

    """

    # Each worker will process results for one (segment, snapshot_dt) pair
    df_preds_lst = []
    for s in peer_indices_dict:
        for snp_dt in peer_indices_dict[s]:
            _df_preds_segment_snapshot = df_preds[
                                                (df_preds[SNAPSHOT_DT_COLUMN] == snp_dt) & 
                                                (df_preds[SEGMENT_COLUMN] == s)
                                            ]
            if(_df_preds_segment_snapshot.shape[0] > 0):
                df_preds_lst.append([
                        xgb_classifier_mdl, 
                        ClassifierLabel,
                        ClassifierLabel_best_outcome,
                        ClassifierLabel_worst_outcome,
                        _df_preds_segment_snapshot,
                        FEATURE_COLUMNS,
                        dim_attr_dict,
                        peer_indices_dict,
                        ID_COLUMN ,
                        SNAPSHOT_DT_COLUMN,
                        SEGMENT_COLUMN,
                        persist_results
                    ])
    results_df_lst = []
    # Use sequential mode if n_parallel <=1 
    if(n_parallel <=1):
        logger.info('n_parallel <=1, falling back to sequential mode')
        results_df_lst = itertools.starmap(compute_dimensional_influence_sequential, df_preds_lst)
    else:
        logger.info('Multiprocessing pool size: {}'.format(n_parallel))
        logger.info('Multiprocessing no. tasks: {}'.format(len(df_preds_lst)))
        # Refer to my comments on https://github.com/dmlc/xgboost/issues/2163
        # If we don't use 'forkserver' xgboost.predict() will hang (since it itself uses OpenMP)
        # for parallel scoring
        forkserver = multiprocessing.get_context('forkserver')
        with forkserver.Pool(n_parallel) as pool:
            results_df_lst = pool.starmap(compute_dimensional_influence_sequential, df_preds_lst)
            pool.close()
            pool.join()

    dim_influence_df = pd.concat(results_df_lst)
    return dim_influence_df

def compute_dimscores_from_diminfluence(
        dim_influence_df,
        ClassifierLabel,
        ClassifierLabel_best_outcome,
        ClassifierLabel_worst_outcome,
        dim_attr_dict,
        SNAPSHOT_DT_COLUMN = 'SNAPSHOT_DT'
    ):
    """Compute dimensional scores (0-100, the higher the better) from dimensional influence

    Parameters
    -----------
    dim_influence_df : Pandas dataframe
        The dataframe obtained as the output from the function ``compute_dimensional_influence_sequential``
        or ``compute_dimensional_influence_parallel``
    ClassifierLabel : 
        An Enum denoting the class labels the classifier is trained on. For example :: 
        
            class ClassifierLabel(Enum):
                GROWTH = 0
                RETENTION = 1
                PARTIAL_ATTR = 2
                TOTAL_ATTR = 3

    ClassifierLabel_best_outcome : enum 'ClassifierLabel'
        Amongst the labels used in the multi-class classifier, what is the best outcome?
        Example : ``ClassifierLabel.GROWTH``
    ClassifierLabel_worst_outcome : enum 'ClassifierLabel'
        Amongst the labels used in the multi-class classifier, what is the worst outcome?
        Example : ``ClassifierLabel.TOTAL_ATTR``
    dim_attr_dict : dict
        A dict of dimension to attribute list mapping. This is obtained as one of the outputs 
        from the function ``preprocess_compute_dimensional_influence()``
    SNAPSHOT_DT_COLUMN : string, default : ``SNAPSHOT_DT``
        The column in ``dim_influence_df`` that corresponds to a the snapshot date. 
        Dimensional score is normalized with respect to ``SNAPSHOT_DT_COLUMN`` segment to capture seasonality.

    Returns
    --------
    dim_influence_df : Pandas dataframe
        The input dataframe with a dimensional score column appended for each dimensional influence column
        in the input.
    min_max_inf_df : Pandas dataframe
        The minimum and maximum values of the 1-D projection (PCA) of dimensional influence for each
        each dimension in the input dataset. This is useful if you choose not to use the dimensional score
        (which is a percentile) but only the dimensional influence 
    dim_inf_dir_df : Pandas dataframe
        The slope of the 1-D projections (y-axis) of the dimensional influence scores against the best outcome
        (x-axis). This is used in determining what value is the 0th percentile and what is the 100th percentile.

    """
    DIMENSIONS = list(dim_attr_dict.keys())
    DIMENSIONAL_INFLUENCE = ['{}_INF'.format(d) for d in DIMENSIONS]
    SNAPSHOT_DATES = dim_influence_df[SNAPSHOT_DT_COLUMN].unique()
    best_and_worst_dim_inf_1d = []
    proj1d_dir_lst = []

    for dim in DIMENSIONAL_INFLUENCE:
        dim_best_outcome = '{}_{}'.format(dim, ClassifierLabel_best_outcome.name)
        dim_worst_outcome = '{}_{}'.format(dim, ClassifierLabel_worst_outcome.name)
        dim_inf_label = '{}_PROJ_1D'.format(dim)
        dim_score_label = '{}_{}'.format(
                                dim.replace('_INF',''), 
                                'SCORE'
                            )
        dim_inf_series_lst = []
        dim_score_series_lst = []
        for snp_dt in SNAPSHOT_DATES:
            inf_2d = dim_influence_df[dim_influence_df[SNAPSHOT_DT_COLUMN] == snp_dt][
                            [
                                dim_best_outcome, 
                                dim_worst_outcome
                            ]
                        ]
            pca = sklearn.decomposition.PCA(n_components = 1)
            pca.fit(inf_2d.values)
            # 1) Project 2d points into line of best fit (first principal component)
            dim_inf = pd.Series(
                            # 1-d projection of the points
                            pca.transform(inf_2d.values)[:, 0],
                            index = dim_influence_df[
                                            dim_influence_df[SNAPSHOT_DT_COLUMN] == snp_dt
                                        ].index
                        )
            # 2) Collect all the 1-D projections in a list
            #    we will concat these in the end and add it as a new column to dim_influence_df
            dim_inf_series_lst.append(dim_inf)  
            # 3) Compute directionality of the 1-D projection w.r.t best outcome
            slope, intercept, rvalue, pvalue, stderr = stats.linregress(
                                                            dim_influence_df[
                                                                    dim_influence_df[SNAPSHOT_DT_COLUMN] == snp_dt
                                                                ][dim_best_outcome], 
                                                                dim_inf
                                                        )
            proj1d_dir_lst.append(
                    [
                        snp_dt,
                        dim,
                        slope,
                        intercept,
                        rvalue,
                        pvalue,
                        stderr,
                        np.sign(slope)
                    ]
                )
            # 4) 1d projection of an 'ideal account' that has p(best_outcome) = 1.0, p(worst_outcome) = 0.0
            best_inf = pca.transform([[1.0, 0.0]])[:, 0][0]
            # 5) 1d projection of an 'worst account' that has p(best_outcome) = 0.0, p(worst_outcome) = 1.0
            worst_inf = pca.transform([[0.0, 1.0]])[:, 0][0]
            best_and_worst_dim_inf_1d.append([snp_dt, dim, best_inf, worst_inf])
            # 6) Compute dimensional score
            dim_inf_sorted = np.sort(dim_inf.values)
            # ptiles will contain values of dim_scores_df[dim] that correspond to the 0th, 1th, ...100th percentiles
            ptiles = np.array([np.percentile(dim_inf_sorted, i) for i in range(101)])
            ptile_func = None
            if (np.sign(slope) < 0):
                # slope is negative
                ptile_func = lambda inf: 100 - np.where(ptiles >= inf)[0][0]
            else:
                # slope is positive
                ptile_func = lambda inf: np.where(ptiles >= inf)[0][0]
            # Convert influence to percentile score
            dim_score_series_lst.append(dim_inf.apply(ptile_func))


        # Add the dimensional influence (1-D projection) for the dimension into the dataframe
        dim_influence_df[dim_inf_label] = pd.concat(dim_inf_series_lst)
        # Add the dimensional score (percentile) for the dimension into the dataframe
        dim_influence_df[dim_score_label] = pd.concat(dim_score_series_lst)

    # Best and worst influence scores for each dimension for each snapshot_dt
    min_max_inf_df = pd.DataFrame(
                            best_and_worst_dim_inf_1d,
                            columns = [
                                        SNAPSHOT_DT_COLUMN,
                                        'DIMENSION', 
                                        'BEST_INF_SCORE', 
                                        'WORST_INF_SCORE'
                                    ]
                        )

    # Directionality of 1-D projections
    dim_inf_dir_df =  pd.DataFrame(
                            proj1d_dir_lst,
                            columns = [
                                        SNAPSHOT_DT_COLUMN,
                                        'DIMENSION', 
                                        'SLOPE', 
                                        'INTERCEPT', 
                                        'RVALUE', 
                                        'PVALUE', 
                                        'STDERR',
                                        'DIRECTION'
                                    ]
                        )

    return dim_influence_df, min_max_inf_df, dim_inf_dir_df

def feat_dir(x_scan, yp, element_id):
    """Get directionality of influence based on a supplied x-axis values and corresponding y-axis values
    
    Parameters
    -----------

    x_scan : list 
        A list of values corresponding to the x-axis
    yp : list
        A list of y-axis values corresponding to the points in ``x_scan``
    element_id : string
        An ID field (ex: an ``ACCOUNT_ID`` or an ``OPTY_ID``) for which the directionality of the feature influence 
        is being computed.

    Returns
    --------
    inf_direction: string 
        The directionality of the influence of the feature. One of ::

            'positive'
            'negative'
            'depends on interaction w/ other attributes'

    """
    slope, intercept, r_value, p_value, std_err = stats.linregress(x_scan, yp)
    inf_direction = None
    if slope > 0:
        inf_direction = 'positive'
    elif slope == 0:
        inf_direction = 'depends on interaction w/ other attributes'
    else:
        inf_direction = 'negative'
    return (element_id, inf_direction)

def compute_attribute_influence_and_directionality(
        df,
        xgb_classifier_mdl, 
        ClassifierLabel,
        ClassifierLabel_outcome,
        FEATURE_COLUMNS,
        n_parallel = int(multiprocessing.cpu_count()*0.50)
    ):
    """Compute partial dependency measure directionality of feature influence in non-linear models

    Based on https://github.com/dmlc/xgboost/issues/1514
    Plot directionality of influence of the attributes using Partial Dependency Plots

    Parameters:
    ------------
    df : Pandas dataframe 
        The input dataframe containing ``FEATURE_COLUMNS``
    xgb_classifier_mdl : XGBClassifier model
        A multi-class or binary XGBClassifier model
    ClassifierLabel : class that extends enum.Enum
        An Enum denoting the class labels the classifier is trained on . For example :: 
        
            class ClassifierLabel(Enum):
                GROWTH = 0
                RETENTION = 1
                PARTIAL_ATTR = 2
                TOTAL_ATTR = 3

    ClassifierLabel_outcome : enum 'ClassifierLabel'
        Amongst the labels used in the multi-class classifier, what is the outcome for 
        which the partial dependency plots are desired?
        Example: ``ClassifierLabel.TOTAL_ATTR``
    FEATURE_COLUMNS : list
        A list of features in the order the input matrix was trained. Ensure that this is
        in sorted order i.e. your ``xgb_classifier_mdl`` should be trained using the same
        columns in the same order.
    n_parallel = int, default : int(multiprocessing.cpu_count()*0.50)
        The number of workers to run the processing on (dependency on number of CPU cores available)

    Returns
    --------
    FEAT_DIRECTION_DICT: dict
        The partial dependency measure. This is a dict where each key corresponds to the name of the attribute
        and the value is a dict composed of the following keys :: 

            influence_score : The strength of this attribute (higher the better)
            influence_rank : The ranking of this attribute's strength (from 0 - len(FEATURE_COLUMNS))
            x_scan : A list of uniformly sampled values for the attribute between it's min and max value
            y_partial : The partial dependency (avg predicted probability) w.r.t to ClassifierLabel_outcome
                        corresponding to the value in x_scan
            inf_direction : A linear approximation of the directionality (positive/negative/neutral) of 
                            influence of the attribute by computing the slope of y_partial w.r.t x_scan.
    """
    # If the input dataframe has > SUBSAMPLE_THRESHOLD rows
    # Then take a sample of the input such that we work with atmost 100K rows
    SUBSAMPLE_THRESHOLD = 100000
    FEAT_DIRECTION_DICT = {}
    # Create a dict holding mapping of feature to it's index, the max & minimum values of this feature
    FEATURES_DICT = {}
    finf_dict = {}
    for i, el in enumerate(
                    sorted(
                        zip(
                            FEATURE_COLUMNS, 
                            xgb_classifier_mdl.feature_importances_
                        ),
                        key = itemgetter(1),
                        reverse = True
                    )
                ):
        # el[0] is the name of the feature 
        # el[1] is the influence score of the feature (higher the better)
        # i is the influence rank (as finf_sorted is sorted by descending order of influence)
        finf_dict[el[0]] = [i, el[1]]

    for i, f in enumerate(FEATURE_COLUMNS):
        feature_vals = df[f].values
        feature_vals = feature_vals[~np.isnan(feature_vals)]
        FEATURES_DICT[f] = {
                                'index': i, 
                                'f_min': np.min(feature_vals), 
                                'f_max': np.max(feature_vals),
                                'influence_score': finf_dict[f][1],
                                'influence_rank': finf_dict[f][0]
                            }
    # End features dict computation
    y_partial = []
    fdir_inp = []
    # Take a 50% sample to speed things up
    X_sample = df.sample(
                    frac = 1.0 if df.shape[0] <= SUBSAMPLE_THRESHOLD else SUBSAMPLE_THRESHOLD/df.shape[0], 
                    replace = False
                )[FEATURE_COLUMNS].values

    step = 0
    for fname in FEATURE_COLUMNS:
        step += 1
        logger.info('{} % complete'.format(int(step*100.0/len(FEATURE_COLUMNS))))
        X_temp = X_sample.copy()
        f_id = FEATURES_DICT[fname]['index']
        f_min = FEATURES_DICT[fname]['f_min'] 
        f_max = FEATURES_DICT[fname]['f_max']
        influence_score = FEATURES_DICT[fname]['influence_score']
        influence_rank = FEATURES_DICT[fname]['influence_rank']
        # Draw 10 uniformly spaced samples between f_min and f_max
        x_scan = np.linspace(f_min, f_max, 10)
        y_partial = []
        for point in x_scan:
            X_temp[:, f_id] = point
            avg_pred_proba_classes = np.mean(
                                            xgb_classifier_mdl.predict_proba(
                                                    pd.DataFrame(
                                                            X_temp,
                                                            columns = FEATURE_COLUMNS
                                                        )
                                                ), 
                                            axis = 0
                                        )
            y_partial.append(avg_pred_proba_classes[ClassifierLabel_outcome.value])

        fdir_inp.append([x_scan, y_partial, fname])
        FEAT_DIRECTION_DICT[fname] = {
                                        # feature influence score
                                        'influence_score': influence_score,
                                        # feature influence rank
                                        'influence_rank': influence_rank,
                                        # range of values for the attribute
                                        'x_scan': x_scan, 
                                        # partial dependence scores for the corresponding x-values
                                        'y_partial': y_partial
                                    }
    inf_direction_lst = []
    # Use sequential mode if n_parallel <=1 
    if(n_parallel <=1):
        logger.info('n_parallel <=1, falling back to sequential mode')
        inf_direction_lst = itertools.starmap(feat_dir, fdir_inp)
    else:
        logger.info('Multiprocessing pool size: {}'.format(n_parallel))
        logger.info('Multiprocessing no. tasks: {}'.format(len(fdir_inp)))
        # Refer to my comments on https://github.com/dmlc/xgboost/issues/2163
        # If we don't use 'forkserver' xgboost.predict() will hang (since it itself uses OpenMP)
        # for parallel scoring
        forkserver = multiprocessing.get_context('forkserver')
        with forkserver.Pool(n_parallel) as pool:
            inf_direction_lst = pool.starmap(feat_dir, fdir_inp)
            pool.close()
            pool.join()

    inf_direction_dict = dict(inf_direction_lst)

    feat_properties_lst = []
    for fname in FEAT_DIRECTION_DICT:
        FEAT_DIRECTION_DICT[fname]['inf_direction'] = inf_direction_dict[fname]
        feat_properties_lst.append([
                            fname, 
                            FEAT_DIRECTION_DICT[fname]['influence_score'],
                            FEAT_DIRECTION_DICT[fname]['influence_rank'], # sort key
                            FEAT_DIRECTION_DICT[fname]['x_scan'],
                            FEAT_DIRECTION_DICT[fname]['y_partial'],
                            FEAT_DIRECTION_DICT[fname]['inf_direction']
                        ])
    return FEAT_DIRECTION_DICT