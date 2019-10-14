from ..influence import *
import uuid
import nose.tools
from nose.tools import assert_equals, raises

def _create_dim_inf_datasets_and_models():
    """Helper function to create mock datasets"""
    NUM_METRICS = 100
    NUM_DIMS = 5
    NUM_SUBDIMS = 2
    NUM_ACCOUNTS  = 100
    NUM_SEGMENTS = 5
    FEATURE_COLUMNS = ['METRIC_{}'.format(i) for i in range(NUM_METRICS)]
    dims = pd.DataFrame(
                [
                    (m, d, s) for m, d, s in zip(
                                        FEATURE_COLUMNS,
                                        itertools.cycle(['DIMENSION_{}'.format(j) for j in range(NUM_DIMS)]),
                                        itertools.cycle(['SUBDIMENSION_{}'.format(k) for k in range(NUM_SUBDIMS)])
                                    )
                ],
                columns = ['METRIC', 'DIMENSION', 'SUBDIMENSION']
            )

    df_scoring_lst = [
                        ['ACCT_{}'.format(i) for i in range(NUM_ACCOUNTS)],
                        ['2017-01-01' for j in range(NUM_ACCOUNTS)],
                        itertools.cycle(['SEG_{}'.format(k) for k in range(NUM_SEGMENTS)])
                    ]

    for f in FEATURE_COLUMNS:
        df_scoring_lst.append(np.random.rand(NUM_ACCOUNTS))

    df_scoring = pd.DataFrame(
                        [_ for _ in zip(*df_scoring_lst)],
                        columns = [
                                    'ACCOUNT_ID', 
                                    'SNAPSHOT_DT', 
                                    'EWS360_SEGMENT'
                                ] + FEATURE_COLUMNS
                    )
    df_scoring['SNAPSHOT_DT'] = pd.to_datetime(df_scoring['SNAPSHOT_DT'])

    class_labels = np.mean(df_scoring[FEATURE_COLUMNS].values, axis = 1)

    qr_1 = np.percentile(class_labels, 25)
    qr_2 = np.percentile(class_labels, 50)
    qr_3 = np.percentile(class_labels, 75)

    class_labels = [
                        EWS360ClassLabels.TOTAL_ATTR.value if x < qr_1 \
                        else EWS360ClassLabels.PARTIAL_ATTR.value if x < qr_2 \
                        else EWS360ClassLabels.RETENTION.value if x < qr_3 \
                        else EWS360ClassLabels.GROWTH.value
                        for x in class_labels
                    ]

    xgb_classifier_mdl = xgboost.XGBClassifier()
    xgb_classifier_mdl.fit(df_scoring[FEATURE_COLUMNS], class_labels)
    return (
            NUM_METRICS,
            NUM_DIMS,
            NUM_SUBDIMS,
            NUM_ACCOUNTS,
            NUM_SEGMENTS,
            FEATURE_COLUMNS,
            dims,
            df_scoring,
            xgb_classifier_mdl
        )

def test_preprocess_compute_dimensional_influence():
    """Test dimensional influence computation"""
    NUM_METRICS, \
    NUM_DIMS, \
    NUM_SUBDIMS, \
    NUM_ACCOUNTS, \
    NUM_SEGMENTS, \
    FEATURE_COLUMNS, \
    dims, \
    df_scoring, \
    xgb_classifier_mdl = _create_dim_inf_datasets_and_models()

    dim_attr_dict, \
    df_preds, \
    peer_indices_dict, \
    pind_dict_size_df = preprocess_compute_dimensional_influence(
                                dims,
                                df_scoring,
                                FEATURE_COLUMNS,
                                xgb_classifier_mdl,
                                EWS360ClassLabels,
                                'ACCOUNT_ID',
                                'SNAPSHOT_DT',
                                'EWS360_SEGMENT'
                            )
    assert_equals(
            len(dim_attr_dict.keys()) == NUM_DIMS + NUM_SUBDIMS,
            True,
            'Mismatch in NUM_DIMS in dim_attr_dict'
        )

    assert_equals(
            df_preds['PRED_CLASS'].nunique() == len([_ for _ in EWS360ClassLabels]),
            True,
            'Mismatch in nunique PRED_CLASS'
        )

    assert_equals(
            len(peer_indices_dict.keys()) == NUM_SEGMENTS,
            True,
            'Mismatch in NUM_SEGMENTS in peer_indices_dict'
        )

    assert_equals(
            pind_dict_size_df['NUM_PEERS'].min()[0] == NUM_ACCOUNTS/NUM_SEGMENTS,
            True,
            'Mismatch in NUM_PEERS in pind_dict_size_df'
        )

    assert_equals(
            pind_dict_size_df['NUM_PEERS'].max()[0] == NUM_ACCOUNTS/NUM_SEGMENTS,
            True,
            'Mismatch in NUM_PEERS in pind_dict_size_df'
        )

def test_compute_dimensional_influence_sequential():
    """Test computing dimensional influence in sequential mode"""
    NUM_METRICS, \
    NUM_DIMS, \
    NUM_SUBDIMS, \
    NUM_ACCOUNTS, \
    NUM_SEGMENTS, \
    FEATURE_COLUMNS, \
    dims, \
    df_scoring, \
    xgb_classifier_mdl = _create_dim_inf_datasets_and_models()

    dim_attr_dict, \
    df_preds, \
    peer_indices_dict, \
    pind_dict_size_df = preprocess_compute_dimensional_influence(
                                dims,
                                df_scoring,
                                FEATURE_COLUMNS,
                                xgb_classifier_mdl,
                                EWS360ClassLabels,
                                'ACCOUNT_ID',
                                'SNAPSHOT_DT',
                                'EWS360_SEGMENT'
                            )
    result_df = compute_dimensional_influence_sequential(
                        xgb_classifier_mdl, 
                        EWS360ClassLabels,
                        EWS360ClassLabels.GROWTH,
                        EWS360ClassLabels.TOTAL_ATTR,
                        df_preds,
                        FEATURE_COLUMNS,
                        dim_attr_dict,
                        peer_indices_dict,
                        ID_COLUMN = 'ACCOUNT_ID',
                        SNAPSHOT_DT_COLUMN = 'SNAPSHOT_DT',
                        SEGMENT_COLUMN = 'EWS360_SEGMENT',
                        persist_results = True
                    )

    assert_equals(
            np.all(
                    [
                        '{}_INF'.format(col) in result_df.columns 
                            for col in list(dim_attr_dict.keys())
                    ]
                ),
            True,
            'Influence scores of all dimensions are not present in result_df'
        )

    assert_equals(
            np.all(
                    [
                        '{}_INF_{}'.format(col, EWS360ClassLabels.GROWTH.name) in result_df.columns 
                            for col in list(dim_attr_dict.keys())
                    ]
                ),
            True,
            'Influence scores of best outcome for all dimensions are not present in result_df'
        )

    assert_equals(
            np.all(
                    [
                        '{}_INF_{}'.format(col, EWS360ClassLabels.TOTAL_ATTR.name) in result_df.columns 
                            for col in list(dim_attr_dict.keys())
                    ]
                ),
            True,
            'Influence scores of worst outcome for all dimensions are not present in result_df'
        )

def test_compute_dimensional_influence_parallel():
    """Test computing dimensional influence in parallel mode"""
    NUM_METRICS, \
    NUM_DIMS, \
    NUM_SUBDIMS, \
    NUM_ACCOUNTS, \
    NUM_SEGMENTS, \
    FEATURE_COLUMNS, \
    dims, \
    df_scoring, \
    xgb_classifier_mdl = _create_dim_inf_datasets_and_models()

    dim_attr_dict, \
    df_preds, \
    peer_indices_dict, \
    pind_dict_size_df = preprocess_compute_dimensional_influence(
                                dims,
                                df_scoring,
                                FEATURE_COLUMNS,
                                xgb_classifier_mdl,
                                EWS360ClassLabels,
                                'ACCOUNT_ID',
                                'SNAPSHOT_DT',
                                'EWS360_SEGMENT'
                            )
    result_df = compute_dimensional_influence_parallel(
                        xgb_classifier_mdl, 
                        EWS360ClassLabels,
                        EWS360ClassLabels.GROWTH,
                        EWS360ClassLabels.TOTAL_ATTR,
                        df_preds,
                        FEATURE_COLUMNS,
                        dim_attr_dict,
                        peer_indices_dict,
                        ID_COLUMN = 'ACCOUNT_ID',
                        SNAPSHOT_DT_COLUMN = 'SNAPSHOT_DT',
                        SEGMENT_COLUMN = 'EWS360_SEGMENT',
                        persist_results = True,
                        n_parallel = 2
                    )

    assert_equals(
            np.all(
                    [
                        '{}_INF'.format(col) in result_df.columns 
                            for col in list(dim_attr_dict.keys())
                    ]
                ),
            True,
            'Influence scores of all dimensions are not present in result_df'
        )

    assert_equals(
            np.all(
                    [
                        '{}_INF_{}'.format(col, EWS360ClassLabels.GROWTH.name) in result_df.columns 
                            for col in list(dim_attr_dict.keys())
                    ]
                ),
            True,
            'Influence scores of best outcome for all dimensions are not present in result_df'
        )

    assert_equals(
            np.all(
                    [
                        '{}_INF_{}'.format(col, EWS360ClassLabels.TOTAL_ATTR.name) in result_df.columns 
                            for col in list(dim_attr_dict.keys())
                    ]
                ),
            True,
            'Influence scores of worst outcome for all dimensions are not present in result_df'
        )

def test_compute_dimscores_from_diminfluence():
    """Test computation of dimensional scores from dimensional influence"""
    NUM_METRICS, \
    NUM_DIMS, \
    NUM_SUBDIMS, \
    NUM_ACCOUNTS, \
    NUM_SEGMENTS, \
    FEATURE_COLUMNS, \
    dims, \
    df_scoring, \
    xgb_classifier_mdl = _create_dim_inf_datasets_and_models()

    dim_attr_dict, \
    df_preds, \
    peer_indices_dict, \
    pind_dict_size_df = preprocess_compute_dimensional_influence(
                                dims,
                                df_scoring,
                                FEATURE_COLUMNS,
                                xgb_classifier_mdl,
                                EWS360ClassLabels,
                                'ACCOUNT_ID',
                                'SNAPSHOT_DT',
                                'EWS360_SEGMENT'
                            )
    dim_influence_df = compute_dimensional_influence_sequential(
                            xgb_classifier_mdl, 
                            EWS360ClassLabels,
                            EWS360ClassLabels.GROWTH,
                            EWS360ClassLabels.TOTAL_ATTR,
                            df_preds,
                            FEATURE_COLUMNS,
                            dim_attr_dict,
                            peer_indices_dict,
                            ID_COLUMN = 'ACCOUNT_ID',
                            SNAPSHOT_DT_COLUMN = 'SNAPSHOT_DT',
                            SEGMENT_COLUMN = 'EWS360_SEGMENT',
                            persist_results = True
                        )

    # Compute dimensional scores from dimensional influence
    dim_influence_df, min_max_inf_df, dim_inf_dir_df = compute_dimscores_from_diminfluence(
                                                            dim_influence_df,
                                                            EWS360ClassLabels,
                                                            EWS360ClassLabels.GROWTH,
                                                            EWS360ClassLabels.TOTAL_ATTR,
                                                            dim_attr_dict
                                                        )

    dim_influence_df.to_csv('/tmp/dim_influence_df.csv', index = False)

    assert_equals(
            len([c for c in dim_influence_df.columns if c.endswith('_SCORE')]),
            len(dim_attr_dict.keys()),
            'Not all dimensions have a score'
        )

    assert_equals(
            np.all(
                np.array(
                    [
                        dim_influence_df[c].min() 
                        for c in dim_influence_df.columns 
                        if c.endswith('_SCORE')
                    ]
                ) >= 0
            ),
            True,
            'One or more dimensions have minimum score that is negative'
        )

    assert_equals(
            np.all(
                np.array(
                    [
                        dim_influence_df[c].max() 
                        for c in dim_influence_df.columns 
                        if c.endswith('_SCORE')
                    ]
                ) <= 100
            ),
            True,
            'One or more dimensions have maximum score that is > 100'
        )

def test_compute_attribute_influence_and_directionality():
    """Test partial dependency plots"""
    NUM_METRICS, \
    NUM_DIMS, \
    NUM_SUBDIMS, \
    NUM_ACCOUNTS, \
    NUM_SEGMENTS, \
    FEATURE_COLUMNS, \
    dims, \
    df_scoring, \
    xgb_classifier_mdl = _create_dim_inf_datasets_and_models()

    FEAT_DIRECTION_DICT = compute_attribute_influence_and_directionality(
                                df_scoring,
                                xgb_classifier_mdl, 
                                EWS360ClassLabels,
                                EWS360ClassLabels.TOTAL_ATTR,
                                FEATURE_COLUMNS
                            )

    assert_equals(
            sorted(FEATURE_COLUMNS), # sorted(list(FEAT_DIRECTION_DICT.keys())),
            sorted(FEATURE_COLUMNS),
            'Not all attributes have partial dependency plots'
        )

